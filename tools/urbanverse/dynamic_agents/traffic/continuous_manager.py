#!/usr/bin/env python3
"""Continuous forward-only UrbanVerse traffic for Go2 integration runs."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..vehicles.traffic_geometry import footprint, polygon_clearance, sat_intersects
from ..vehicles.route_planning import STATIC_SAFETY_MARGIN_M, physical_road_height
from ..core.vehicle_motion import (
    apply_junction_signal_control,
    bicycle_step,
    route_pose_at_arc,
)
from .multivehicle_manager import (
    DIVERSE_TRAFFIC_VEHICLE_IDS,
    MIXED_VEHICLE_IDS,
    Scene10MultiVehicleManager,
)
from .routes import load_automotive_routes, orca_longitudinal_caps


class Scene10ContinuousVehicleManager(Scene10MultiVehicleManager):
    """Run the portable automotive lifecycle beside a real-policy Go2.

    The historical class name remains API-compatible, but behavior and scene
    geometry are selected by ``TrafficSceneConfig``.
    """

    continuous_looping = True

    def __init__(
        self,
        *args: Any,
        automotive_routes_path: Path,
        random_seed: int = 20260815,
        initial_fill: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("traffic_asset_ids", DIVERSE_TRAFFIC_VEHICLE_IDS)
        super().__init__(*args, **kwargs)
        if self.traffic_vehicle_count < 1:
            raise ValueError("enabled continuous traffic requires at least one vehicle")
        self.automotive_routes_path = Path(automotive_routes_path).resolve()
        base_routes, base_route_rows = load_automotive_routes(
            self.automotive_routes_path,
            self.lane,
            self.static,
            self.specs,
        )
        base_count = len(base_routes)
        self.routes = [
            base_routes[index % base_count] for index in range(self.traffic_vehicle_count)
        ]
        self.automotive_route_rows = []
        for index in range(self.traffic_vehicle_count):
            row = dict(base_route_rows[index % base_count])
            row.update(
                {
                    "id": index,
                    "base_route_id": index % base_count,
                    "traffic_cohort": index // base_count,
                }
            )
            self.automotive_route_rows.append(row)
        self.route_names = tuple(
            f"{base_route_rows[index % base_count]['route_name']}_flow_{index // base_count:02d}"
            for index in range(self.traffic_vehicle_count)
        )
        self.random = random.Random(int(random_seed))
        self.initial_fill = bool(initial_fill)
        self.junction_group_count = int(
            self.scene_config.traffic.get("junction_group_count", 3)
            if self.scene_config is not None
            else 3
        )
        self.intersection_owner: int | None = 0
        self.intersection_phase = "green"
        self.intersection_phase_started_s = 0.0
        self.intersection_green_duration_s = float(
            self.scene_config.traffic.get("green_duration_s", 12.0)
            if self.scene_config is not None
            else 12.0
        )
        self.intersection_stop_line_m = float(
            self.scene_config.traffic.get("stop_line_m", 10.0)
            if self.scene_config is not None
            else 10.0
        )
        self.intersection_reservation_changes = 0
        self.dynamic_proposal_guard_events = 0
        self.static_proposal_guard_events = 0
        self.boundary_proposal_guard_events = 0
        self.external_yield_steps = 0
        self.external_proposal_guard_events = 0
        self.maximum_simultaneously_moving_vehicle_count = 0
        self.maximum_simultaneously_driving_vehicle_count = 0
        self.moving_vehicle_step_count = 0
        self.driving_vehicle_step_count = 0
        self.update_count = 0
        self.orca_candidate_evaluation_count = 0
        self.profile_wall_s = {
            "spawn": 0.0,
            "orca": 0.0,
            "junction": 0.0,
            "proposal_and_static_guard": 0.0,
            "dynamic_guard_and_commit": 0.0,
            "publish_and_grounding": 0.0,
            "update_total": 0.0,
        }

        junction = np.asarray(
            self.scene_config.traffic.get("junction_xy", [-625.0, 490.0])
            if self.scene_config is not None
            else [-625.0, 490.0],
            dtype=np.float64,
        )
        desired_merge_times = tuple(
            float(value)
            for value in (
                self.scene_config.traffic.get(
                    "desired_merge_times_s", [21.0, 25.0, 23.0, 28.0, 15.0]
                )
                if self.scene_config is not None
                else [21.0, 25.0, 23.0, 28.0, 15.0]
            )
        )
        if not desired_merge_times:
            raise ValueError("traffic desired_merge_times_s must not be empty")
        self.agents = []
        for index, (spec, route) in enumerate(zip(self.specs, self.routes)):
            route_slot = index % base_count
            cohort = index // base_count
            route_row = base_route_rows[route_slot]
            junction_group = int(route_row.get("junction_group", 0 if route_slot in (0, 1) else 1 if route_slot in (2, 3) else 2))
            route_junction = np.asarray(route_row.get("junction_xy", junction), dtype=np.float64)
            nearest = int(np.argmin(np.linalg.norm(route.xy - route_junction, axis=1)))
            desired_speed = float(route_row.get("desired_speed_mps", min(4.0, 3.5 + 0.15 * route_slot + 0.03 * cohort)))
            spawn_at = max(
                0.0,
                desired_merge_times[route_slot % len(desired_merge_times)]
                - float(route.arc_m[nearest]) / desired_speed,
                ) + float(route_row.get("cohort_headway_s", 4.0)) * cohort
            wheelbase = float(np.clip(spec["length"] * 0.58, 2.3, 3.5))
            self.agents.append(
                {
                    "id": index,
                    "route_name": self.route_names[index],
                    "route_slot": route_slot,
                    "junction_group": junction_group,
                    "shared_lane_group": str(route_row.get("shared_lane_group", f"route_{route_slot}")),
                    "asset_id": spec["asset_id"],
                    "category": spec["category"],
                    "length": spec["length"],
                    "width": spec["width"],
                    "wheelbase": wheelbase,
                    "minimum_radius": self.automotive_route_rows[index][
                        "minimum_turning_radius_m"
                    ],
                    "orca_radius": 0.5 * math.hypot(spec["length"], spec["width"]) + 0.18,
                    "radius": 0.5 * math.hypot(spec["length"], spec["width"]) + 0.18,
                    "desired_speed": desired_speed,
                    "preferred_speed": desired_speed,
                    "position": route.xy[0].copy(),
                    "heading": float(route.yaw[0]),
                    "velocity": np.zeros(2, dtype=np.float64),
                    "speed": 0.0,
                    "steering": 0.0,
                    "route_arc_m": 0.0,
                    "junction_arc_m": float(route.arc_m[nearest]),
                    "junction_wait_s": 0.0,
                    "status": "waiting",
                    "start_time_s": spawn_at,
                    "next_spawn_time_s": spawn_at,
                    "cycles_completed": 0,
                    "initial_route_arc_m": 0.0,
                    "spawn_count": 0,
                    "wait_reason": "initial_schedule",
                    "maximum_heading_motion_error_deg": 0.0,
                }
            )

        # Each additional appearance must fit the actual route assigned to it.
        # The original cache proved only the five base planning envelopes.
        self.route_validation_sample_count = 0
        for agent, route in zip(self.agents, self.routes):
            for sample_index, (position, heading) in enumerate(zip(route.xy, route.yaw)):
                body = footprint(
                    position,
                    float(heading),
                    agent["length"],
                    agent["width"],
                )
                if not all(self.lane.contains(corner) for corner in body):
                    raise RuntimeError(
                        f"diverse vehicle V{agent['id']} left Lane on route sample {sample_index}"
                    )
                if any(
                    sat_intersects(body, obstacle, margin=STATIC_SAFETY_MARGIN_M)
                    for _, obstacle in self.static
                ):
                    raise RuntimeError(
                        f"diverse vehicle V{agent['id']} hit a static OBB on route sample {sample_index}"
                    )
                self.route_validation_sample_count += 1

        # Portable scenes do not need a whole-scene obstacle map. Instead,
        # sweep the exact fixed traffic corridors against the composed PhysX
        # scene before motion. A ray that lands on a parked vehicle or on a
        # surface far above/below the audited road height rejects the config.
        self.physx_route_preflight_query_count = 0
        self.physx_route_preflight_miss_count = 0
        self.physx_route_preflight_paths: set[str] = set()
        if self.scene_config is not None and bool(
            self.scene_config.traffic.get("physx_route_preflight", True)
        ):
            tolerance = float(
                self.scene_config.traffic.get("road_height_tolerance_m", 0.20)
            )
            interval = float(
                self.scene_config.traffic.get("physx_route_preflight_interval_m", 1.0)
            )
            for agent, route in zip(self.agents, self.routes):
                last_arc = -math.inf
                for sample_index, (arc_m, position, heading) in enumerate(
                    zip(route.arc_m, route.xy, route.yaw)
                ):
                    if arc_m - last_arc + 1.0e-9 < interval and sample_index != len(route.xy) - 1:
                        continue
                    last_arc = float(arc_m)
                    body = footprint(position, float(heading), agent["length"], agent["width"])
                    for query_xy in np.vstack((position, body)):
                        self.physx_route_preflight_query_count += 1
                        try:
                            ground_z, ground_path = physical_road_height(self.scene_query, query_xy)
                        except RuntimeError:
                            self.physx_route_preflight_miss_count += 1
                            if bool(
                                self.scene_config.traffic.get(
                                    "physx_route_preflight_require_hit", True
                                )
                            ):
                                raise RuntimeError(
                                    f"route {agent['route_name']} has no PhysX support at "
                                    f"sample {sample_index}: xy={query_xy.tolist()}"
                                )
                            continue
                        self.physx_route_preflight_paths.add(str(ground_path))
                        if abs(float(ground_z) - self.calibrated_flat_road_z_m) > tolerance:
                            raise RuntimeError(
                                f"route {agent['route_name']} has non-road support at sample "
                                f"{sample_index}: z={ground_z:.3f}, expected="
                                f"{self.calibrated_flat_road_z_m:.3f}, path={ground_path}"
                            )

        self.initially_filled_vehicle_count = 0
        self.loop_passage=None
        if base_count==1 and self.routes[0].closed:
            from ..core.loop_passage import LoopPassage
            self.loop_passage=LoopPassage(self.routes[0],max(a['length'] for a in self.agents),max(a['width'] for a in self.agents))
        if self.initial_fill:
            initial_junction_group = 0
            self.intersection_owner = initial_junction_group
            placed_polygons: list[np.ndarray] = []
            lane_ranks: dict[tuple[str, int], int] = {}
            for agent in self.agents:
                route = self.routes[agent["id"]]
                route_slot = int(agent["route_slot"])
                lane_key = ("shared_lane", str(self.automotive_route_rows[agent["id"]].get("shared_lane_group", f"route_{route_slot}")))
                lane_rank = lane_ranks.get(lane_key, 0)
                lane_ranks[lane_key] = lane_rank + 1
                preferred_arc = 8.0 + 18.0 * lane_rank
                if route.closed:
                    cohort=sum(int(a['route_slot'])==route_slot for a in self.agents)
                    preferred_arc=(lane_rank+.5)*route.length_m/max(1,cohort)
                candidate_arcs = list(
                    np.arange(8.0, max(8.1, route.length_m - 7.9), 8.0)
                )
                if route.closed:candidate_arcs=list(np.arange(0.,route.length_m,2.))
                if int(agent["junction_group"]) != initial_junction_group:
                    stop_arc = float(agent["junction_arc_m"]) - 10.5
                    candidate_arcs = [arc for arc in candidate_arcs if arc <= stop_arc]
                candidate_arcs.sort(key=lambda arc: abs(float(arc) - preferred_arc))
                for candidate_arc in candidate_arcs:
                    if self.loop_passage is not None and self.loop_passage.inside(float(candidate_arc)):
                        continue
                    position, heading = route_pose_at_arc(route, float(candidate_arc))
                    body = footprint(
                        position,
                        heading,
                        agent["length"],
                        agent["width"],
                    )
                    if any(
                        sat_intersects(body, other, margin=0.50)
                        for other in placed_polygons
                    ):
                        continue
                    agent.update(
                        {
                            "status": "moving",
                            "position": position,
                            "heading": float(heading),
                            "velocity": np.zeros(2, dtype=np.float64),
                            "speed": 0.0,
                            "steering": 0.0,
                            "route_arc_m": float(candidate_arc),
                            "initial_route_arc_m": float(candidate_arc),
                            "junction_wait_s": 0.0,
                            "wait_reason": None,
                            "start_time_s": 0.0,
                            "next_spawn_time_s": 0.0,
                            "spawn_count": 1,
                        }
                    )
                    placed_polygons.append(body)
                    self.initially_filled_vehicle_count += 1
                    break

        # NearRoad is deliberately separated from Lane.  The wider evidence
        # window reflects camera visibility, not a collision/avoidance radius.
        self.third_person_longitudinal_limits_m = (-6.0, 24.0)
        self.third_person_lateral_limit_m = 18.0
        self.third_person_range_limit_m = 26.0

    @property
    def all_terminal(self) -> bool:
        # A continuous traffic population has no terminal global state.
        return False

    @property
    def all_arrived(self) -> bool:
        return False

    def _is_agent_active_on_road(self, agent: dict[str, Any]) -> bool:
        return agent["status"] == "moving"

    @staticmethod
    def _copy_agent(agent: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in agent.items()
        }

    def update(
        self,
        timestamp_s: float,
        go2_xy: np.ndarray,
        go2_yaw_rad: float = 0.0,
        yield_obstacles: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        update_started = time.perf_counter()
        phase_started = update_started
        active = [agent for agent in self.agents if agent["status"] == "moving"]
        active_polygons = {
            agent["id"]: footprint(
                agent["position"], agent["heading"], agent["length"], agent["width"]
            )
            for agent in active
        }
        spawn_candidates = sorted(
            self.agents,
            key=lambda agent: (
                float(agent["next_spawn_time_s"]),
                int(agent["spawn_count"]),
                int(agent["id"]),
            ),
        )
        for agent in spawn_candidates:
            if (
                agent["status"] != "waiting"
                or timestamp_s + 1.0e-9 < agent["next_spawn_time_s"]
            ):
                continue
            route = self.routes[agent["id"]]
            spawn_poly = footprint(
                route.xy[0], route.yaw[0], agent["length"], agent["width"]
            )
            blocked = any(
                sat_intersects(spawn_poly, polygon, margin=0.35)
                for polygon in active_polygons.values()
            )
            if blocked:
                agent["wait_reason"] = "spawn_occupied"
                agent["next_spawn_time_s"] = timestamp_s + 0.5
                continue
            agent.update(
                {
                    "status": "moving",
                    "position": route.xy[0].copy(),
                    "heading": float(route.yaw[0]),
                    "velocity": np.zeros(2, dtype=np.float64),
                    "speed": 0.0,
                    "steering": 0.0,
                    "route_arc_m": 0.0,
                    "junction_wait_s": 0.0,
                    "wait_reason": None,
                    "start_time_s": timestamp_s,
                }
            )
            agent["spawn_count"] += 1
            active_polygons[agent["id"]] = spawn_poly

        self.profile_wall_s["spawn"] += time.perf_counter() - phase_started

        active = [agent for agent in self.agents if agent["status"] == "moving"]
        self.maximum_simultaneously_moving_vehicle_count = max(
            self.maximum_simultaneously_moving_vehicle_count,
            len(active),
        )
        phase_started = time.perf_counter()
        speed_caps, orca_metrics = orca_longitudinal_caps(active, self.dt)
        if self.loop_passage is not None:
            self.loop_passage.apply(active,speed_caps,timestamp_s)
        external_rows = [
            (
                np.asarray(item["position_xy"], dtype=np.float64),
                float(item["radius_m"]),
            )
            for item in yield_obstacles
        ]
        for centre, radius in external_rows:
            if centre.shape != (2,) or radius <= 0.0:
                raise ValueError("traffic yield obstacles require xy and positive radius")
        for agent in active:
            forward = np.asarray(
                (math.cos(float(agent["heading"])), math.sin(float(agent["heading"]))),
                dtype=np.float64,
            )
            cap = float(speed_caps.get(agent["id"], agent["desired_speed"]))
            for centre, radius in external_rows:
                relative = centre - agent["position"]
                longitudinal = float(np.dot(relative, forward))
                lateral = abs(float(relative[0] * forward[1] - relative[1] * forward[0]))
                stop_distance = 0.5 * float(agent["length"]) + radius + 0.35
                if longitudinal <= 0.0 or lateral >= 0.5 * float(agent["width"]) + radius + 0.35:
                    continue
                clearance = longitudinal - stop_distance
                external_cap = float(agent["desired_speed"]) * float(
                    np.clip(clearance / 3.0, 0.0, 1.0)
                )
                if external_cap < cap:
                    self.external_yield_steps += 1
                    cap = external_cap
            speed_caps[agent["id"]] = cap
        self.profile_wall_s["orca"] += time.perf_counter() - phase_started
        self.orca_constraint_count += int(orca_metrics["orca_constraint_count"])
        self.orca_predicted_conflict_count += int(orca_metrics["predicted_conflicts"])
        self.orca_candidate_evaluation_count += int(orca_metrics["candidate_evaluations"])
        phase_started = time.perf_counter()
        changed = False
        if (
            self.intersection_phase == "green"
            and timestamp_s - self.intersection_phase_started_s
            >= self.intersection_green_duration_s
        ):
            self.intersection_phase = "clearance"
            changed = True
        if self.intersection_phase == "clearance":
            group_inside = any(
                int(agent["junction_group"]) == int(self.intersection_owner)
                and float(agent["route_arc_m"])
                >= float(agent["junction_arc_m"]) - self.intersection_stop_line_m
                and float(agent["route_arc_m"])
                <= float(agent["junction_arc_m"]) + 18.0
                for agent in active
            )
            if not group_inside:
                self.intersection_owner = (int(self.intersection_owner) + 1) % self.junction_group_count
                self.intersection_phase = "green"
                self.intersection_phase_started_s = float(timestamp_s)
                changed = True
        apply_junction_signal_control(
            active,
            speed_caps,
            self.intersection_owner,
            clearance_phase=self.intersection_phase == "clearance",
            stop_line_m=self.intersection_stop_line_m,
        )
        self.profile_wall_s["junction"] += time.perf_counter() - phase_started
        self.intersection_reservation_changes += int(changed)

        phase_started = time.perf_counter()
        old_states = {agent["id"]: self._copy_agent(agent) for agent in active}
        proposed = {}
        proposed_polys = {}
        rejected: set[int] = set()
        for agent in active:
            maximum_steering = math.atan(agent["wheelbase"] / agent["minimum_radius"])
            if self.scene_config and 'maximum_steering_deg' in self.scene_config.traffic:
                maximum_steering=math.radians(float(self.scene_config.traffic['maximum_steering_deg']))
                if not 0<maximum_steering<=math.radians(40):
                    raise ValueError('Configured automotive steering bound must be in (0,40] degrees')
            candidate = bicycle_step(
                agent,
                self.routes[agent["id"]],
                self.dt,
                agent["desired_speed"],
                speed_caps.get(agent["id"], agent["desired_speed"]),
                agent["wheelbase"],
                maximum_steering,
                lookahead_m=float(self.scene_config.traffic.get('controller_lookahead_m',4.0)) if self.scene_config else 4.0,
                use_target_chord=bool(self.scene_config.traffic.get('use_target_chord',False)) if self.scene_config else False,
                steer_before_drive=bool(self.scene_config.traffic.get('steer_before_drive',False)) if self.scene_config else False,
            )
            proposed[agent["id"]] = candidate
            body = footprint(
                candidate["position"],
                candidate["heading"],
                agent["length"],
                agent["width"],
            )
            proposed_polys[agent["id"]] = body
            if hasattr(self.lane,'pose_clear') and not self.lane.pose_clear(
                candidate['position'],candidate['heading'],agent['length'],agent['width']):
                rejected.add(agent['id'])
                self.boundary_proposal_guard_events+=1
            body_min = body.min(axis=0)
            body_max = body.max(axis=0)
            if not all(self.lane.contains(corner) for corner in body):
                rejected.add(agent["id"])
                self.boundary_proposal_guard_events += 1
            elif any(
                sat_intersects(body, obstacle, margin=STATIC_SAFETY_MARGIN_M)
                for _, obstacle, obstacle_min, obstacle_max in self.static_bounds
                if np.all(body_max + STATIC_SAFETY_MARGIN_M >= obstacle_min)
                and np.all(obstacle_max + STATIC_SAFETY_MARGIN_M >= body_min)
            ):
                rejected.add(agent["id"])
                self.static_proposal_guard_events += 1
            elif any(
                sat_intersects(
                    body,
                    np.asarray(
                        [
                            [centre[0] - radius, centre[1] - radius],
                            [centre[0] + radius, centre[1] - radius],
                            [centre[0] + radius, centre[1] + radius],
                            [centre[0] - radius, centre[1] + radius],
                        ],
                        dtype=np.float64,
                    ),
                    margin=0.12,
                )
                for centre, radius in external_rows
            ):
                rejected.add(agent["id"])
                self.external_proposal_guard_events += 1

        self.profile_wall_s["proposal_and_static_guard"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        for left_index, left in enumerate(active):
            for right in active[left_index + 1 :]:
                if sat_intersects(
                    proposed_polys[left["id"]],
                    proposed_polys[right["id"]],
                    margin=0.12,
                ):
                    same_stream = left["shared_lane_group"] == right["shared_lane_group"]
                    if same_stream:
                        # Preserve traffic flow: stop only the trailing body.
                        # Rejecting both members of a following pair creates a
                        # permanent symmetric standstill even though the leader
                        # has clear road ahead.
                        trailing = (
                            left
                            if float(left["route_arc_m"]) < float(right["route_arc_m"])
                            else right
                        )
                        stream_route=self.routes[left['id']]
                        if stream_route.closed:
                            forward_gap=(float(right['route_arc_m'])-float(left['route_arc_m']))%stream_route.length_m
                            trailing=left if forward_gap<stream_route.length_m*.5 else right
                        rejected.add(trailing["id"])
                    else:
                        rejected.update((left["id"], right["id"]))
                    self.dynamic_proposal_guard_events += 1

        for agent in active:
            agent_id = agent["id"]
            if agent_id in rejected:
                accepted = old_states[agent_id]
                accepted["speed"] = 0.0
                accepted["velocity"] = np.zeros(2, dtype=np.float64)
            else:
                accepted = proposed[agent_id]
            agent.clear()
            agent.update(accepted)
            if agent["speed"] > 0.08:
                motion_heading = math.atan2(
                    float(agent["velocity"][1]), float(agent["velocity"][0])
                )
                error = abs(
                    (motion_heading - float(agent["heading"]) + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
                agent["maximum_heading_motion_error_deg"] = max(
                    agent["maximum_heading_motion_error_deg"], math.degrees(error)
                )
            route = self.routes[agent_id]
            if route.closed:
                agent['cycles_completed']=route.full_laps_since(agent['initial_route_arc_m'],agent['route_arc_m'])
            elif (
                agent["route_arc_m"] >= route.length_m - 2.0
                or np.linalg.norm(agent["position"] - route.xy[-1]) <= 1.8
            ):
                agent["cycles_completed"] += 1
                agent["status"] = "waiting"
                agent["speed"] = 0.0
                agent["velocity"] = np.zeros(2, dtype=np.float64)
                agent["next_spawn_time_s"] = timestamp_s + self.random.uniform(1.5, 5.0)
                agent["wait_reason"] = "random_cooldown"

        driving_count = sum(
            agent["status"] == "moving" and float(agent["speed"]) > 0.25
            for agent in self.agents
        )
        moving_count = sum(agent["status"] == "moving" for agent in self.agents)
        self.maximum_simultaneously_driving_vehicle_count = max(
            self.maximum_simultaneously_driving_vehicle_count,
            driving_count,
        )
        self.moving_vehicle_step_count += moving_count
        self.driving_vehicle_step_count += driving_count
        self.update_count += 1
        self.profile_wall_s["dynamic_guard_and_commit"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        state = self._publish_frame(timestamp_s, go2_xy, go2_yaw_rad)
        self.profile_wall_s["publish_and_grounding"] += time.perf_counter() - phase_started
        self.profile_wall_s["update_total"] += time.perf_counter() - update_started
        state["simultaneously_driving_vehicle_count"] = driving_count
        state["intersection_group"] = self.intersection_owner
        state["intersection_phase"] = self.intersection_phase
        if self.loop_passage is not None:state['loop_passage']=self.loop_passage.summary()
        return state

    def summary(self) -> dict[str, Any]:
        visual_alignment = self.measure_visual_geometry_alignment()
        visual_errors = [
            row["rendered_axis_to_planner_error_deg"]
            for row in visual_alignment
            if row["rendered_axis_to_planner_error_deg"] is not None
        ]
        return {
            "continuous_looping": True,
            "initial_fill_enabled": self.initial_fill,
            "initially_filled_vehicle_count": self.initially_filled_vehicle_count,
            "dynamic_vehicle_count": len(self.agents),
            "moving_vehicle_count_at_end": sum(
                agent["status"] == "moving" for agent in self.agents
            ),
            "maximum_simultaneously_moving_vehicle_count": (
                self.maximum_simultaneously_moving_vehicle_count
            ),
            "maximum_simultaneously_driving_vehicle_count": (
                self.maximum_simultaneously_driving_vehicle_count
            ),
            "mean_moving_vehicle_count": (
                self.moving_vehicle_step_count / self.update_count if self.update_count else 0.0
            ),
            "mean_driving_vehicle_count": (
                self.driving_vehicle_step_count / self.update_count if self.update_count else 0.0
            ),
            "spawn_counts": [agent["spawn_count"] for agent in self.agents],
            "completed_cycles": [agent["cycles_completed"] for agent in self.agents],
            "total_completed_cycles": sum(agent["cycles_completed"] for agent in self.agents),
            "vehicle_outcomes": [
                {
                    "id": agent["id"],
                    "route_name": agent["route_name"],
                    "asset_id": agent["asset_id"],
                    "category": agent["category"],
                    "status": agent["status"],
                    "spawn_count": agent["spawn_count"],
                    "cycles_completed": agent["cycles_completed"],
                    "start_xy": self.routes[agent["id"]].xy[0].tolist(),
                    "goal_xy": self.routes[agent["id"]].xy[-1].tolist(),
                }
                for agent in self.agents
            ],
            "integrated_vehicle_asset_ids": list(self.traffic_asset_ids),
            "traffic_instance_asset_ids": [
                agent["asset_id"] for agent in self.agents
            ],
            "unique_integrated_vehicle_asset_ids": sorted(
                {agent["asset_id"] for agent in self.agents}
            ),
            "excluded_camera_visibility_asset_ids": [MIXED_VEHICLE_IDS[1]],
            "static_vehicle_count": len(self.static),
            "minimum_dynamic_obb_clearance_m": self._finite_or_none(
                self.minimum_dynamic_clearance_m
            ),
            "minimum_static_obb_clearance_m": self._finite_or_none(
                self.minimum_static_clearance_m
            ),
            "minimum_go2_vehicle_clearance_m": self._finite_or_none(
                self.minimum_go2_clearance_m
            ),
            "go2_radius_m": self.go2_radius_m,
            "dynamic_obb_overlap_events": self.dynamic_overlap_events,
            "static_obb_overlap_events": self.static_overlap_events,
            "go2_vehicle_overlap_events": self.go2_vehicle_overlap_events,
            "dynamic_proposal_guard_events": self.dynamic_proposal_guard_events,
            "static_proposal_guard_events": self.static_proposal_guard_events,
            "road_boundary_proposal_guard_events": self.boundary_proposal_guard_events,
            "external_yield_steps": self.external_yield_steps,
            "external_proposal_guard_events": self.external_proposal_guard_events,
            "third_person_dynamic_pass_max_duration_s": self.third_person_pass_max_s,
            "third_person_dynamic_pass_vehicle_ids": sorted(
                self.third_person_pass_vehicle_ids
            ),
            "third_person_dynamic_pass_contract": (
                "moving vehicle centre within chase-camera evidence window: "
                f"longitudinal {list(self.third_person_longitudinal_limits_m)} m, "
                f"|lateral| <= {self.third_person_lateral_limit_m} m, "
                f"range <= {self.third_person_range_limit_m} m"
            ),
            "orca_constraint_count": self.orca_constraint_count,
            "orca_predicted_conflict_count": self.orca_predicted_conflict_count,
            "orca_candidate_evaluation_count": self.orca_candidate_evaluation_count,
            "intersection_reservation_changes": self.intersection_reservation_changes,
            "intersection_reservation_mode": (
                "timed green -> occupied-junction clearance -> next compatible lane group"
            ),
            "intersection_green_duration_s": self.intersection_green_duration_s,
            "intersection_phase_at_end": self.intersection_phase,
            "route_validation_sample_count": self.route_validation_sample_count,
            "physx_route_preflight": {
                "query_count": self.physx_route_preflight_query_count,
                "miss_count": self.physx_route_preflight_miss_count,
                "hit_paths": sorted(self.physx_route_preflight_paths),
                "method": "centre plus four OBB-corner down-rays over every configured route",
            },
            "lane_spatial_index": {
                "cell_size_m": self.lane.spatial_cell_size_m,
                "query_count": self.lane.query_count,
                "candidate_triangle_count": self.lane.candidate_count,
                "mean_candidates_per_query": (
                    self.lane.candidate_count / self.lane.query_count
                    if self.lane.query_count
                    else 0.0
                ),
            },
            "profile_wall_s": dict(self.profile_wall_s),
            "maximum_heading_motion_error_deg": max(
                agent["maximum_heading_motion_error_deg"] for agent in self.agents
            ),
            "maximum_rendered_body_axis_error_deg": max(visual_errors, default=None),
            "route_lengths_m": [route.length_m for route in self.routes],
            "automotive_routes": str(self.automotive_routes_path),
            "lifecycle": (
                "persistent actor -> forward bicycle drive -> continuous closed seam; no despawn or respawn"
                if all(route.closed for route in self.routes) else
                "random cooldown -> safe spawn -> forward bicycle drive -> "
                "endpoint disappearance -> random cooldown"
            ),
            "motion_model": (
                "kinematic bicycle; CPU ORCA supplies longitudinal speed caps only; "
                "timed compatible traffic-signal groups serialize conflicting junction paths"
            ),
            "go2_interaction": (
                "none in this stage; Go2 follows a spatially separated NearRoad route"
            ),
            "ground_height": {
                "runtime_raycast_enabled_by_config": bool(
                    self.scene_config.traffic.get("runtime_ground_raycast", True)
                    if self.scene_config is not None
                    else True
                ),
                "raycast_hit_count": self.ground_raycast_hit_count,
                "raycast_reuse_count": self.ground_raycast_reuse_count,
                "query_interval_s": self.ground_query_interval_s,
                "query_distance_m": self.ground_query_distance_m,
                "calibrated_fallback_count": self.ground_calibrated_fallback_count,
                "fallback_z_m": self.calibrated_flat_road_z_m,
            },
            "visibility": {
                "runtime_visual_clones": self.runtime_visual_clones,
                "enforcement_count": self.visibility_enforcement_count,
                "readbacks": self.visibility_readbacks,
            },
            "visual_geometry_alignment": visual_alignment,
            "method": (
                "validated automotive route cache + kinematic bicycle + CPU ORCA "
                "spatial-neighbour longitudinal caps + grouped junction reservation + "
                "OBB guards + scene-configured runtime road grounding"
            ),
            "collision_evidence": (
                "2D oriented vehicle bodies against dynamic/static vehicle bodies and Lane"
            ),
        }
