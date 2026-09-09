"""Project-controlled resident People roaming on approved raster regions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .official_people import OfficialPeopleManager
from .roaming import RoamingAssignment
from .walkable_regions import WalkableRegions


def build_project_people_payload(
    regions: WalkableRegions,
    assignments: Sequence[RoamingAssignment],
    assets_root: Path,
    people_settings: Mapping[str, Any],
    *,
    simulation_dt_s: float = 0.02,
) -> dict[str, Any]:
    """Build the in-memory payload used to preauthor NVIDIA People skins."""

    from ..micromobility.roaming import plan_block_path

    root = Path(assets_root).resolve()
    walk_animation = root / "Animations/stand_walk_loop_in_place.skelanim.usd"
    retarget_source = root / "Characters/Biped_Setup.usd"
    for required in (walk_animation, retarget_source):
        if not required.is_file():
            raise FileNotFoundError(required)
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        start = np.asarray(assignment.start_xyz[:2], dtype=np.float64)
        target = np.asarray(assignment.initial_target_xyz[:2], dtype=np.float64)
        closed_path=people_settings.get('closed_paths_xy',{}).get(assignment.name)
        path = np.asarray(closed_path,float) if closed_path is not None else plan_block_path(
            regions,
            assignment.component_id,
            start,
            target,
            spacing_m=float(people_settings.get("path_spacing_m", 0.75)),
        )
        rows.append(
            {
                "name": assignment.name,
                "character_usd": str(assignment.asset.resolve()),
                "spawn_xyz": list(map(float, assignment.start_xyz)),
                "route_start_xyz": list(map(float, assignment.start_xyz)),
                "route_points_xyz": [
                    [float(point[0]), float(point[1]), regions.config.ground_z_m]
                    for point in path[1:]
                ],
                "preferred_speed_mps": float(
                    people_settings.get("preferred_speed_mps", 1.0)
                ),
                "initial_active": True,
                "component_id": int(assignment.component_id),
                "closed_loop": closed_path is not None,
            }
        )
    return {
        "schema_version": 1,
        "assets_root": str(root),
        "walk_animation_usd": str(walk_animation),
        "retarget_source_character_usd": str(retarget_source),
        "character_root_prim": "/World/ground/terrain/Characters",
        "simulation_dt_s": float(simulation_dt_s),
        "random_seed": int(people_settings.get("seed", regions.config.fixed_seed)),
        "animation_rate_time_codes_per_s": float(
            people_settings.get("animation_rate_time_codes_per_s", 30.0)
        ),
        "animation_update_hz": float(people_settings.get("animation_update_hz", 25.0)),
        "lifecycle_mode": "resident_random_roaming",
        "dynamic_avoidance_enabled": True,
        "pedestrians": rows,
    }


class ProjectPeopleRoamingManager(OfficialPeopleManager):
    """Move resident People roots along seeded A* tasks without NVIDIA navigation."""

    def __init__(
        self,
        stage: Any,
        config: dict[str, Any],
        run_dir: Path,
        regions: WalkableRegions,
        assignments: Sequence[RoamingAssignment],
        *,
        seed: int,
        arrival_radius_m: float = 0.55,
        minimum_trip_m: float = 4.0,
        maximum_trip_m: float = 25.0,
        pedestrian_radius_m: float = 0.30,
        hard_clearance_m: float = 0.08,
        stall_retarget_after_s: float = 6.0,
        path_spacing_m: float = 0.75,
        raster_admission_tolerance_m: float = 0.35,
    ) -> None:
        super().__init__(stage, config, run_dir)
        if [agent["name"] for agent in self.agents] != [item.name for item in assignments]:
            raise ValueError("project People assignment order does not match authored roots")
        self.regions = regions
        self.assignments = tuple(assignments)
        self.rng = np.random.default_rng(seed)
        self.arrival_radius_m = float(arrival_radius_m)
        self.minimum_trip_m = float(minimum_trip_m)
        self.maximum_trip_m = float(maximum_trip_m)
        self.pedestrian_radius_m = float(pedestrian_radius_m)
        self.hard_clearance_m = float(hard_clearance_m)
        self.stall_retarget_after_s = float(stall_retarget_after_s)
        self.path_spacing_m = float(path_spacing_m)
        self.raster_admission_tolerance_m = float(raster_admission_tolerance_m)
        if self.maximum_trip_m <= self.minimum_trip_m:
            raise ValueError("maximum People trip must exceed minimum trip")
        if self.pedestrian_radius_m <= 0.0 or self.hard_clearance_m < 0.0:
            raise ValueError("People footprint parameters are invalid")
        if self.raster_admission_tolerance_m < 0.0:
            raise ValueError("People raster admission tolerance must be nonnegative")
        self.completed_trips = {item.name: 0 for item in assignments}
        self.retarget_failures = {item.name: 0 for item in assignments}
        self.hard_stop_steps = {item.name: 0 for item in assignments}
        self.raster_tolerance_admitted_steps = {
            item.name: 0 for item in assignments
        }
        self.shared_avoidance_limited_steps = {
            item.name: 0 for item in assignments
        }
        self.stall_retargets = {item.name: 0 for item in assignments}
        self.stall_elapsed_s = {item.name: 0.0 for item in assignments}
        self.targets: dict[str, np.ndarray] = {}
        self.current_velocities: dict[str, np.ndarray] = {}
        for index, (agent, assignment) in enumerate(zip(self.agents, assignments)):
            agent["component_id"] = int(assignment.component_id)
            agent['closed_loop']=bool(config['pedestrians'][index].get('closed_loop',False))
            agent["route"] = np.asarray(
                [config["pedestrians"][index]["route_start_xyz"], *config["pedestrians"][index]["route_points_xyz"]],
                dtype=np.float64,
            )
            agent["waypoint"] = min(1, len(agent["route"]) - 1)
            agent["direction"] = 1
            self.targets[agent["name"]] = agent["route"][-1, :2].copy()
            self.current_velocities[agent["name"]] = np.zeros(2, dtype=np.float64)

    def positions(self) -> np.ndarray:
        return np.asarray([agent["position"] for agent in self.agents], dtype=np.float64)

    def mixed_states(self) -> tuple[Any, ...]:
        from ..core.mixed_avoidance import MixedAgentState

        return tuple(
            MixedAgentState(
                agent_id=agent["name"],
                kind="pedestrian",
                position_xy=agent["position"][:2],
                velocity_xy=self.current_velocities[agent["name"]],
                radius_m=self.pedestrian_radius_m,
            )
            for agent in self.agents
        )

    def _plan_task(self, index: int, occupied_xy: np.ndarray) -> np.ndarray:
        from ..micromobility.roaming import plan_block_path

        agent = self.agents[index]
        start = agent["position"][:2].copy()
        component_id = int(agent["component_id"])
        for _ in range(1024):
            target = self.regions.sample(component_id, self.rng)
            distance = float(np.linalg.norm(target - start))
            if not self.minimum_trip_m <= distance <= self.maximum_trip_m:
                continue
            if len(occupied_xy) and float(np.linalg.norm(occupied_xy - target, axis=1).min()) < 1.0:
                continue
            try:
                path = plan_block_path(
                    self.regions,
                    component_id,
                    start,
                    target,
                    spacing_m=self.path_spacing_m,
                )
            except (ValueError, RuntimeError):
                continue
            if len(path) >= 2:
                return np.column_stack(
                    (path, np.full(len(path), self.regions.config.ground_z_m))
                )
        raise RuntimeError(f"could not sample project People task for {agent['name']}")

    @staticmethod
    def _external_speed_scale(
        position_xy: np.ndarray,
        direction_xy: np.ndarray,
        obstacles: Sequence[tuple[np.ndarray, float]],
    ) -> float:
        scale = 1.0
        for obstacle_xy, combined_radius in obstacles:
            relative = np.asarray(obstacle_xy, dtype=np.float64) - position_xy
            longitudinal = float(np.dot(relative, direction_xy))
            lateral = abs(float(direction_xy[0] * relative[1] - direction_xy[1] * relative[0]))
            if longitudinal <= 0.0 or lateral >= combined_radius + 0.5:
                continue
            clearance = float(np.linalg.norm(relative)) - combined_radius
            scale = min(scale, float(np.clip((clearance - 0.10) / 1.5, 0.0, 1.0)))
        return scale

    def prepare_step(
        self,
        traffic_agents: list[dict[str, Any]] | None,
        go2_xy: np.ndarray,
        go2_velocity_xy: np.ndarray,
        *,
        micromobility_states: Mapping[str, Any] | None = None,
        micromobility_specs: Sequence[Any] = (),
        micromobility_catalog: Mapping[str, Any] | None = None,
        shared_speed_scales: Mapping[str, float] | None = None,
        ignore_go2: bool = False,
    ) -> None:
        """Advance A* tasks and apply the shared reciprocal safety decision."""

        del go2_velocity_xy
        self.ignore_go2 = bool(ignore_go2)
        positions = self.positions()
        occupied = positions[:, :2]
        for index, agent in enumerate(self.agents):
            route = agent["route"]
            while agent["waypoint"] < len(route) and float(
                np.linalg.norm(route[agent["waypoint"], :2] - agent["position"][:2])
            ) <= self.arrival_radius_m:
                agent["waypoint"] += 1
                agent["waypoint_transition_count"] += 1
            if agent["waypoint"] < len(route):
                continue
            self.completed_trips[agent["name"]] += 1
            if agent.get('closed_loop'):
                agent['waypoint']=1
                continue
            try:
                route = self._plan_task(index, np.delete(occupied, index, axis=0))
            except RuntimeError:
                self.retarget_failures[agent["name"]] += 1
                agent["waypoint"] = len(agent["route"]) - 1
                continue
            agent["route"] = route
            agent["waypoint"] = 1
            self.targets[agent["name"]] = route[-1, :2].copy()

        desired_directions: list[np.ndarray] = []
        desired_speeds: list[float] = []
        traffic_obstacles = [
            (
                np.asarray(item["position"], dtype=np.float64),
                float(item.get("radius", 1.5)) + self.pedestrian_radius_m,
            )
            for item in (traffic_agents or [])
            if item.get("status") in ("moving", "stopped")
        ]
        go2_obstacles = (
            []
            if self.ignore_go2
            else [
                (
                    np.asarray(go2_xy, dtype=np.float64),
                    0.67 + self.pedestrian_radius_m,
                )
            ]
        )
        micro_states = dict(micromobility_states or {})
        micro_specs_by_id = {item.agent_id: item for item in micromobility_specs}
        micro_catalog = dict(micromobility_catalog or {})
        micro_obstacles = []
        for agent_id, state in micro_states.items():
            spec = micro_specs_by_id.get(agent_id)
            calibration = micro_catalog.get(spec.asset_id) if spec is not None else None
            radius = (
                math.hypot(calibration.length_m * 0.5, calibration.width_m * 0.5)
                if calibration is not None
                else 0.8
            )
            micro_obstacles.append((state.position_xy, radius + self.pedestrian_radius_m))
        for agent in self.agents:
            target = agent["route"][min(agent["waypoint"], len(agent["route"]) - 1)]
            delta = target[:2] - agent["position"][:2]
            distance = float(np.linalg.norm(delta))
            direction = delta / max(distance, 1.0e-9)
            desired_directions.append(direction)
            external_scale = self._external_speed_scale(
                agent["position"][:2],
                direction,
                [*traffic_obstacles, *go2_obstacles, *micro_obstacles],
            )
            shared_scale = float((shared_speed_scales or {}).get(agent["name"], 1.0))
            if shared_scale < 0.999:
                self.shared_avoidance_limited_steps[agent["name"]] += 1
            desired_speeds.append(float(agent["preferred_speed"]) * min(external_scale, shared_scale))

        candidates = [agent["position"][:2].copy() for agent in self.agents]
        for index, (agent, direction, speed) in enumerate(
            zip(self.agents, desired_directions, desired_speeds)
        ):
            distance_to_waypoint = float(
                np.linalg.norm(
                    agent["route"][min(agent["waypoint"], len(agent["route"]) - 1), :2]
                    - agent["position"][:2]
                )
            )
            candidates[index] = agent["position"][:2] + direction * min(
                distance_to_waypoint, speed * self.dt
            )

        rejected: set[int] = set()
        minimum_pair_distance = 2.0 * self.pedestrian_radius_m + self.hard_clearance_m
        for first in range(len(candidates)):
            component_id = int(self.agents[first]["component_id"])
            if self.regions.component_at(candidates[first]) != component_id:
                if self.regions.in_component_with_raster_tolerance(
                    candidates[first],
                    component_id,
                    self.raster_admission_tolerance_m,
                ):
                    self.raster_tolerance_admitted_steps[
                        self.agents[first]["name"]
                    ] += 1
                else:
                    rejected.add(first)
            for second in range(first + 1, len(candidates)):
                if float(np.linalg.norm(candidates[first] - candidates[second])) < minimum_pair_distance:
                    rejected.update((first, second))
        for index, candidate in enumerate(candidates):
            if index in rejected:
                continue
            if any(
                float(np.linalg.norm(candidate - centre)) < radius + self.hard_clearance_m
                for centre, radius in [
                    *traffic_obstacles,
                    *go2_obstacles,
                    *micro_obstacles,
                ]
            ):
                rejected.add(index)

        moving_count = 0
        for index, (agent, candidate, direction, requested_speed) in enumerate(
            zip(self.agents, candidates, desired_directions, desired_speeds)
        ):
            name = agent["name"]
            previous = agent["position"][:2].copy()
            if index in rejected:
                candidate = previous
                requested_speed = 0.0
                self.hard_stop_steps[name] += 1
            displacement = candidate - previous
            distance = float(np.linalg.norm(displacement))
            if distance > 1.0e-9:
                agent["heading"] = math.atan2(float(direction[1]), float(direction[0]))
                moving_count += 1
                self.stall_elapsed_s[name] = 0.0
            else:
                self.stall_elapsed_s[name] += self.dt
            agent["position"][:2] = candidate
            agent["speed"] = distance / self.dt
            self.current_velocities[name] = displacement / self.dt
            self.path_lengths[name] += distance
            self._author_pose(index, agent["position"], agent["heading"])
            self._advance_animation_clock(index)
            if self.update_count % self.animation_update_stride == 0:
                self._apply_animation_sample(index)
            if self.stall_elapsed_s[name] >= self.stall_retarget_after_s and not agent.get('closed_loop'):
                try:
                    agent["route"] = self._plan_task(index, np.delete(occupied, index, axis=0))
                    agent["waypoint"] = 1
                    self.targets[name] = agent["route"][-1, :2].copy()
                    self.stall_retargets[name] += 1
                except RuntimeError:
                    self.retarget_failures[name] += 1
                self.stall_elapsed_s[name] = 0.0
        self.maximum_simultaneously_moving = max(self.maximum_simultaneously_moving, moving_count)
        self.maximum_active_pedestrian_count = len(self.agents)
        self.simulation_time_s += self.dt

    def summary(self) -> dict[str, Any]:
        return {
            "implementation": "project A* resident roaming + reciprocal mixed-agent speed limits + UsdSkel walk animation",
            "pedestrian_count": len(self.agents),
            "resident_count": len(self.agents),
            "position_source": "project-controlled USD root transform",
            "root_motion_controller": "project A* paths inside approved connected component",
            "navmesh_enabled": False,
            "character_manager_enabled": False,
            "navigation_manager_enabled": False,
            "dynamic_obstacle_enabled": False,
            "go2_interaction": (
                "ignored by pedestrian motion and avoidance"
                if getattr(self, "ignore_go2", False)
                else "treated as an external avoidance obstacle"
            ),
            "component_ids": [int(agent["component_id"]) for agent in self.agents],
            "completed_trips": self.completed_trips,
            "retarget_failures": self.retarget_failures,
            "stall_retargets": self.stall_retargets,
            "hard_stop_steps": self.hard_stop_steps,
            "raster_admission_tolerance_m": self.raster_admission_tolerance_m,
            "raster_tolerance_admitted_steps": (
                self.raster_tolerance_admitted_steps
            ),
            "shared_avoidance_limited_steps": (
                self.shared_avoidance_limited_steps
            ),
            "path_lengths_m": self.path_lengths,
            "maximum_simultaneously_moving_pedestrian_count": self.maximum_simultaneously_moving,
            "active_pedestrian_count": len(self.agents),
            "maximum_active_pedestrian_count": self.maximum_active_pedestrian_count,
            "final_positions_xyz": {agent["name"]: agent["position"].tolist() for agent in self.agents},
            "current_targets_xy": {name: target.tolist() for name, target in self.targets.items()},
            "walk_animation_usd": self.config["walk_animation_usd"],
            "animation_clock_mode": "actual root speed scaled; frozen at stop",
            "animation_stop_threshold_mps": self.animation_stop_threshold_mps,
            "animation_paused_steps": self.animation_paused_steps,
            "animation_advanced_steps": self.animation_advanced_steps,
            "visual_bboxes": self._visual_bboxes(),
            "update_count": self.update_count,
        }
