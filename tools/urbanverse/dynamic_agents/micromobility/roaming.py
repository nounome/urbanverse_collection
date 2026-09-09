"""Seeded block tasks and constrained bicycle tracking for micromobility."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

from ..pedestrians.walkable_regions import WalkableRegions, _label_components
from .calibration import AssetCalibration
from .avoidance import polygons_intersect
from .motion import (
    MicromobilityState,
    MotionLimits,
    footprint_corners,
    step_bicycle,
    wrap_angle,
)


@dataclass(frozen=True)
class MicromobilityAgentSpec:
    agent_id: str
    asset_id: str
    component_id: int
    desired_speed_mps: float


@dataclass
class MicromobilityTask:
    path_xy: np.ndarray
    target_xy: np.ndarray
    waypoint_index: int = 1


def _pixels_support_trip(
    pixels: np.ndarray,
    pixel_size_xy_m: Sequence[float],
    minimum_trip_m: float,
) -> bool:
    """Cheap conservative diameter test for one connected safe island."""

    if len(pixels) < 2:
        return False
    xy = np.column_stack(
        (
            pixels[:, 1] * float(pixel_size_xy_m[0]),
            pixels[:, 0] * float(pixel_size_xy_m[1]),
        )
    )
    projections = (xy[:, 0], xy[:, 1], xy[:, 0] + xy[:, 1], xy[:, 0] - xy[:, 1])
    extreme_indices = {
        int(index)
        for values in projections
        for index in (np.argmin(values), np.argmax(values))
    }
    extremes = xy[sorted(extreme_indices)]
    diameter = float(
        np.linalg.norm(extremes[:, None, :] - extremes[None, :, :], axis=2).max()
    )
    return diameter + min(map(float, pixel_size_xy_m)) >= float(minimum_trip_m)


def eligible_micromobility_components(
    regions: WalkableRegions,
    catalog: Mapping[str, AssetCalibration],
    *,
    margin_m: float = 0.15,
    minimum_safe_pixels: int = 16,
    minimum_trip_m: float = 0.0,
) -> tuple[int, ...]:
    """Return blocks wide enough for every catalog footprint orientation.

    The conservative enclosing radius makes the admission independent of
    current yaw and prevents assigning a two-wheeler to a pedestrian-only
    sliver.  Runtime still checks the exact OBB on every step.
    """
    if not catalog:
        raise ValueError("micromobility catalog must be nonempty")
    radius_m = max(
        math.hypot(item.length_m * 0.5, item.width_m * 0.5) + margin_m
        for item in catalog.values()
    )
    radius_px = int(math.ceil(radius_m / min(regions.pixel_size_xy_m)))
    filter_size = 2 * radius_px + 1
    eligible: list[int] = []
    for component_id in regions.component_ids:
        mask = (regions.labels == component_id).astype(np.uint8) * 255
        eroded = np.asarray(
            Image.fromarray(mask).filter(ImageFilter.MinFilter(filter_size))
        ) > 0
        eroded_labels, eroded_sizes = _label_components(eroded)
        has_task_island = any(
            size >= int(minimum_safe_pixels)
            and _pixels_support_trip(
                np.argwhere(eroded_labels == island),
                regions.pixel_size_xy_m,
                minimum_trip_m,
            )
            for island, size in enumerate(eroded_sizes, start=1)
        )
        if has_task_island:
            eligible.append(int(component_id))
    if not eligible:
        raise RuntimeError("no walkable component admits the calibrated micromobility catalog")
    return tuple(eligible)


def balanced_component_assignment(
    component_ids: Sequence[int],
    weights: Sequence[float],
    *,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Seeded weighted assignment with a portable per-block density cap.

    Pure weighted sampling put three or more two-wheelers into one narrow
    NearRoad component while leaving another qualified block empty.  Fill at
    most ``ceil(count / block_count)`` slots per block, retaining area weights
    among blocks that still have capacity.  The rule is scene-independent and
    only uses components that passed the footprint admission gate.
    """

    ids = np.asarray(component_ids, dtype=np.int32)
    probabilities = np.asarray(weights, dtype=np.float64)
    if count <= 0 or len(ids) == 0 or probabilities.shape != ids.shape:
        raise ValueError("component ids, weights and count must be nonempty")
    if np.any(probabilities < 0.0) or float(probabilities.sum()) <= 0.0:
        raise ValueError("component weights must be nonnegative with positive sum")
    capacity = int(math.ceil(int(count) / len(ids)))
    used = np.zeros(len(ids), dtype=np.int32)
    assigned: list[int] = []
    for _ in range(int(count)):
        available = used < capacity
        choices = np.flatnonzero(available)
        local = probabilities[choices]
        local /= local.sum()
        selected = int(rng.choice(choices, p=local))
        used[selected] += 1
        assigned.append(int(ids[selected]))
    return np.asarray(assigned, dtype=np.int32)


def _astar_pixels(labels: np.ndarray, component_id: int, start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    if int(labels[start]) != component_id or int(labels[goal]) != component_id:
        raise ValueError("A* endpoints are outside the requested connected component")
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    height, width = labels.shape
    while queue:
        _estimate, current_cost, current = heapq.heappop(queue)
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return np.asarray(path[::-1], dtype=np.int32)
        if current_cost > cost.get(current, math.inf):
            continue
        for dy, dx, step_cost in neighbours:
            candidate = (current[0] + dy, current[1] + dx)
            if not (0 <= candidate[0] < height and 0 <= candidate[1] < width):
                continue
            if int(labels[candidate]) != component_id:
                continue
            candidate_cost = current_cost + step_cost
            if candidate_cost >= cost.get(candidate, math.inf):
                continue
            cost[candidate] = candidate_cost
            parent[candidate] = current
            heuristic = math.hypot(goal[0] - candidate[0], goal[1] - candidate[1])
            heapq.heappush(queue, (candidate_cost + heuristic, candidate_cost, candidate))
    raise RuntimeError(f"no grid path in connected component {component_id}")


def _line_pixels(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return every raster cell crossed by a segment between cell centres.

    Rounding one sample per dominant-axis cell can miss a cell crossed near a
    boundary.  The later metric resampling then walks through a cell that the
    simplifier never checked.  Traverse grid boundaries directly so the
    visibility test and runtime ``world_to_pixel`` membership agree.
    """

    row, column = map(int, first)
    end_row, end_column = map(int, second)
    delta_row = end_row - row
    delta_column = end_column - column
    step_row = int(np.sign(delta_row))
    step_column = int(np.sign(delta_column))
    time_delta_row = 1.0 / abs(delta_row) if delta_row else math.inf
    time_delta_column = 1.0 / abs(delta_column) if delta_column else math.inf
    time_to_row_boundary = 0.5 / abs(delta_row) if delta_row else math.inf
    time_to_column_boundary = (
        0.5 / abs(delta_column) if delta_column else math.inf
    )
    crossed = [(row, column)]
    while (row, column) != (end_row, end_column):
        if time_to_row_boundary < time_to_column_boundary:
            row += step_row
            time_to_row_boundary += time_delta_row
        elif time_to_column_boundary < time_to_row_boundary:
            column += step_column
            time_to_column_boundary += time_delta_column
        else:
            row += step_row
            column += step_column
            time_to_row_boundary += time_delta_row
            time_to_column_boundary += time_delta_column
        crossed.append((row, column))
    return np.asarray(crossed, dtype=np.int32)


def _simplify_pixels(path: np.ndarray, labels: np.ndarray, component_id: int) -> np.ndarray:
    retained = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1:
            line = _line_pixels(path[anchor], path[candidate])
            if np.all(labels[line[:, 0], line[:, 1]] == component_id):
                break
            candidate -= 1
        retained.append(path[candidate])
        anchor = candidate
    return np.asarray(retained, dtype=np.int32)


def plan_block_path(
    regions: WalkableRegions,
    component_id: int,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    spacing_m: float = 0.75,
    center_admission_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Plan a centre path inside a connected, footprint-safe raster.

    The base region is inflated for pedestrians.  Bikes and mopeds have a
    larger swept radius, so callers can provide a further-eroded raster.  This
    prevents A* from selecting a legal centre line whose exact runtime OBB can
    never negotiate the adjacent curb or obstacle boundary.
    """
    labels = regions.labels if center_admission_labels is None else center_admission_labels
    if labels.shape != regions.labels.shape:
        raise ValueError("center admission raster shape mismatch")
    start = regions.world_to_pixel(start_xy)
    target = regions.world_to_pixel(target_xy)
    if start is None or target is None:
        raise ValueError("path endpoints are outside the region raster")
    pixels = _simplify_pixels(
        _astar_pixels(labels, component_id, start, target),
        labels,
        component_id,
    )
    points = np.asarray([regions.pixel_to_world(int(row), int(column)) for row, column in pixels])
    dense: list[np.ndarray] = [points[0]]
    for first, second in zip(points, points[1:]):
        distance = float(np.linalg.norm(second - first))
        for ratio in np.linspace(0.0, 1.0, max(2, int(math.ceil(distance / spacing_m)) + 1))[1:]:
            dense.append(first * (1.0 - ratio) + second * ratio)
    return np.asarray(dense, dtype=np.float64)


class MicromobilityRoamingManager:
    """Random block tasks with non-holonomic tracking and hard emergency stops."""

    def __init__(
        self,
        regions: WalkableRegions,
        specs: Sequence[MicromobilityAgentSpec],
        catalog: dict[str, AssetCalibration],
        *,
        seed: int,
        minimum_trip_m: float = 10.0,
        maximum_trip_m: float | None = None,
        initial_endpoints_xy: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
        excluded_spawn_xy: Sequence[Sequence[float]] = (),
        closed_paths_xy: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.regions = regions
        self.specs = tuple(specs)
        self.catalog = catalog
        self.rng = np.random.default_rng(seed)
        self.minimum_trip_m = float(minimum_trip_m)
        self.maximum_trip_m = (
            None if maximum_trip_m is None else float(maximum_trip_m)
        )
        if (
            self.maximum_trip_m is not None
            and self.maximum_trip_m <= self.minimum_trip_m
        ):
            raise ValueError("maximum_trip_m must be greater than minimum_trip_m")
        self.states: dict[str, MicromobilityState] = {}
        self.tasks: dict[str, MicromobilityTask] = {}
        self.closed_paths_xy=dict(closed_paths_xy or {})
        self.closed_progress={spec.agent_id:0. for spec in specs}
        self.closed_routes={}
        self.completed_tasks = {spec.agent_id: 0 for spec in specs}
        self.emergency_stops = {spec.agent_id: 0 for spec in specs}
        self.consecutive_emergency_stop_s = {spec.agent_id: 0.0 for spec in specs}
        self.stall_replans = {spec.agent_id: 0 for spec in specs}
        self.planning_start_projections = {spec.agent_id: 0 for spec in specs}
        self.maximum_planning_start_projection_m = {
            spec.agent_id: 0.0 for spec in specs
        }
        self.stall_replan_after_s = 5.0
        self.shared_avoidance_limited_steps = {spec.agent_id: 0 for spec in specs}
        self.external_emergency_stop_steps = {spec.agent_id: 0 for spec in specs}
        self._center_admission_labels: dict[str, np.ndarray] = {}
        self._center_admission_pixels: dict[tuple[str, int], np.ndarray] = {}
        for asset_id, calibration in catalog.items():
            if self.closed_paths_xy:
                continue  # No random task sampling or isotropic extra erosion.
            swept_radius_m = math.hypot(
                calibration.length_m * 0.5, calibration.width_m * 0.5
            ) + 0.15
            radius_px = int(
                math.ceil(swept_radius_m / min(regions.pixel_size_xy_m))
            )
            safe_labels = np.zeros_like(regions.labels)
            next_safe_component = 1
            for component_id in regions.component_ids:
                mask = (regions.labels == component_id).astype(np.uint8) * 255
                eroded = np.asarray(
                    Image.fromarray(mask).filter(
                        ImageFilter.MinFilter(2 * radius_px + 1)
                    )
                ) > 0
                # Footprint erosion can split one pedestrian component into
                # multiple disconnected vehicle-centre islands.  Give those
                # islands distinct planning labels so task sampling can never
                # choose endpoints separated by a curb/obstacle bottleneck.
                eroded_labels, eroded_sizes = _label_components(eroded)
                admitted_islands: list[np.ndarray] = []
                for eroded_component in range(1, len(eroded_sizes) + 1):
                    island_pixels = np.argwhere(
                        eroded_labels == eroded_component
                    )
                    if not _pixels_support_trip(
                        island_pixels,
                        regions.pixel_size_xy_m,
                        self.minimum_trip_m,
                    ):
                        continue
                    safe_labels[eroded_labels == eroded_component] = (
                        next_safe_component
                    )
                    admitted_islands.append(island_pixels)
                    next_safe_component += 1
                if admitted_islands:
                    self._center_admission_pixels[
                        (asset_id, int(component_id))
                    ] = np.concatenate(admitted_islands, axis=0)
            self._center_admission_labels[asset_id] = safe_labels
        occupied: list[np.ndarray] = [
            np.asarray(point, dtype=np.float64) for point in excluded_spawn_xy
        ]
        if any(point.shape != (2,) for point in occupied):
            raise ValueError("excluded_spawn_xy entries must be xy points")
        component_initial_headings: dict[int, float] = {}
        for spec in specs:
            endpoint_override = (initial_endpoints_xy or {}).get(spec.agent_id)
            if self.closed_paths_xy:
                from ..core.vehicle_motion import AutomotiveRoute,cumulative_lengths
                path=np.asarray(self.closed_paths_xy[spec.agent_id],float)
                if not self.regions.contains_points(path,spec.component_id):raise ValueError('Closed micro path leaves component')
                if np.linalg.norm(path[0]-path[-1])>1e-6:raise ValueError('Micro path is not closed')
                delta=np.roll(path[:-1],-1,axis=0)-np.roll(path[:-1],1,axis=0)
                headings=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
                headings=np.r_[headings,headings[-1]+wrap_angle(headings[0]-headings[-1])]
                self.closed_routes[spec.agent_id]=AutomotiveRoute(path,headings,cumulative_lengths(path),catalog[spec.asset_id].minimum_turning_radius_m,True)
                start=path[0];task=MicromobilityTask(path,path[-1])
                if any(np.linalg.norm(start-other)<2.5 for other in occupied):raise ValueError('Closed micro spawn spacing is insufficient')
            elif endpoint_override is None:
                start, task = self._sample_safe_task(
                    spec,
                    occupied,
                    preferred_heading_rad=component_initial_headings.get(
                        spec.component_id
                    ),
                )
            else:
                requested_start = np.asarray(endpoint_override[0], dtype=np.float64)
                requested_target = np.asarray(endpoint_override[1], dtype=np.float64)
                if requested_start.shape != (2,) or requested_target.shape != (2,):
                    raise ValueError(
                        f"initial endpoints for {spec.agent_id} must be two xy points"
                    )
                if (
                    self.regions.component_at(requested_start) != spec.component_id
                    or self.regions.component_at(requested_target) != spec.component_id
                ):
                    raise ValueError(
                        f"initial endpoints for {spec.agent_id} leave component "
                        f"{spec.component_id}"
                    )
                if np.linalg.norm(requested_target - requested_start) < self.minimum_trip_m:
                    raise ValueError(
                        f"initial endpoints for {spec.agent_id} are shorter than "
                        f"{self.minimum_trip_m:.2f} m"
                    )
                start, task = self._repair_reviewed_task(
                    spec, requested_start, requested_target, occupied
                )
            occupied.append(start)
            path = task.path_xy
            heading = self._path_heading(path, at_start=True)
            component_initial_headings.setdefault(spec.component_id, heading)
            self.states[spec.agent_id] = MicromobilityState(float(start[0]), float(start[1]), heading)
            if spec.agent_id in self.closed_routes:
                route=self.closed_routes[spec.agent_id]
                k=wrap_angle(route.yaw[1]-route.yaw[0])/max(route.arc_m[1],1e-6)
                steering=float(np.clip(math.atan(catalog[spec.asset_id].wheelbase_m*k),
                    -catalog[spec.asset_id].maximum_steering_rad,catalog[spec.asset_id].maximum_steering_rad))
                self.states[spec.agent_id]=replace(self.states[spec.agent_id],yaw_rad=float(route.yaw[0]),steering_rad=steering)
            self.tasks[spec.agent_id] = task

    def _closed_lookahead(self,spec,state):
        from ..core.vehicle_motion import nearest_forward_arc,route_pose_at_arc
        route=self.closed_routes[spec.agent_id]
        arc=nearest_forward_arc(route,state.position_xy,self.closed_progress[spec.agent_id])
        self.closed_progress[spec.agent_id]=arc
        self.completed_tasks[spec.agent_id]=int(arc//route.length_m)
        return route_pose_at_arc(route,arc+1.5)[0]

    @staticmethod
    def _path_heading(path: np.ndarray, *, at_start: bool) -> float:
        first, second = (path[0], path[1]) if at_start else (path[-2], path[-1])
        return math.atan2(float(second[1] - first[1]), float(second[0] - first[0]))

    def _task_endpoints_safe(
        self,
        spec: MicromobilityAgentSpec,
        path: np.ndarray,
    ) -> bool:
        calibration = self.catalog[spec.asset_id]
        for point, at_start in ((path[0], True), (path[-1], False)):
            state = MicromobilityState(
                float(point[0]), float(point[1]), self._path_heading(path, at_start=at_start)
            )
            if not self.regions.contains_points(
                footprint_corners(state, calibration, margin_m=0.15), spec.component_id
            ):
                return False
        return True

    def _task_length_safe(self, path: np.ndarray) -> bool:
        if self.maximum_trip_m is None:
            return True
        path_length_m = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        return path_length_m <= self.maximum_trip_m

    def _plan_path(
        self,
        spec: MicromobilityAgentSpec,
        start_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> np.ndarray:
        safe_labels = self._center_admission_labels[spec.asset_id]
        start_pixel = self.regions.world_to_pixel(start_xy)
        target_pixel = self.regions.world_to_pixel(target_xy)
        if start_pixel is None or target_pixel is None:
            raise ValueError("micromobility task endpoint is outside the region raster")
        safe_component = int(safe_labels[start_pixel])
        if safe_component == 0 or int(safe_labels[target_pixel]) != safe_component:
            raise RuntimeError(
                "micromobility task endpoints are in different footprint-safe islands"
            )
        return plan_block_path(
            self.regions,
            safe_component,
            start_xy,
            target_xy,
            center_admission_labels=safe_labels,
        )

    def _sample_center(self, spec: MicromobilityAgentSpec) -> np.ndarray:
        pixels = self._center_admission_pixels.get((spec.asset_id, spec.component_id))
        if pixels is None or not len(pixels):
            raise RuntimeError(
                f"component {spec.component_id} has no footprint-safe centres "
                f"for {spec.asset_id}"
            )
        row, column = pixels[int(self.rng.integers(0, len(pixels)))]
        return self.regions.pixel_to_world(int(row), int(column))

    def _sample_safe_task(
        self,
        spec: MicromobilityAgentSpec,
        occupied: Sequence[np.ndarray],
        *,
        preferred_heading_rad: float | None = None,
    ) -> tuple[np.ndarray, MicromobilityTask]:
        for _ in range(1024):
            start = self._sample_separated(spec, occupied, 2.5)
            target = self._sample_target(spec, start)
            path = self._plan_path(spec, start, target)
            heading_ok = (
                preferred_heading_rad is None
                or abs(
                    wrap_angle(
                        self._path_heading(path, at_start=True)
                        - preferred_heading_rad
                    )
                )
                <= math.radians(75.0)
            )
            if (
                heading_ok
                and self._task_length_safe(path)
                and self._task_endpoints_safe(spec, path)
            ):
                return start, MicromobilityTask(path, target)
        raise RuntimeError(f"could not sample footprint-safe task for {spec.agent_id}")

    def _repair_reviewed_task(
        self,
        spec: MicromobilityAgentSpec,
        requested_start: np.ndarray,
        requested_target: np.ndarray,
        occupied: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, MicromobilityTask]:
        """Keep reviewed endpoints when safe, otherwise find nearby admitted ones.

        Review maps often contain centre-line points intended for pedestrians.
        A bicycle is longer than a pedestrian radius, so using those centres
        verbatim can put its nose outside the approved block.  Candidate
        repairs remain in the same component and within 6 m of the reviewed
        endpoints; this preserves the intended interaction rather than silently
        relocating the agent elsewhere in the scene.
        """
        for attempt in range(2048):
            start = (
                requested_start
                if attempt == 0
                else self._sample_center(spec)
            )
            target = (
                requested_target
                if attempt == 0
                else self._sample_center(spec)
            )
            if (
                np.linalg.norm(start - requested_start) > 6.0
                or np.linalg.norm(target - requested_target) > 6.0
                or np.linalg.norm(target - start) < self.minimum_trip_m
                or (
                    self.maximum_trip_m is not None
                    and np.linalg.norm(target - start) > self.maximum_trip_m
                )
                or any(np.linalg.norm(start - point) < 1.0 for point in occupied)
            ):
                continue
            try:
                path = self._plan_path(spec, start, target)
            except (ValueError, RuntimeError):
                continue
            if self._task_length_safe(path) and self._task_endpoints_safe(spec, path):
                return start, MicromobilityTask(path, target)
        raise RuntimeError(f"could not repair reviewed footprint endpoints for {spec.agent_id}")

    def _sample_separated(
        self,
        spec: MicromobilityAgentSpec,
        occupied: Sequence[np.ndarray],
        distance_m: float,
    ) -> np.ndarray:
        for _ in range(512):
            candidate = self._sample_center(spec)
            if all(np.linalg.norm(candidate - point) >= distance_m for point in occupied):
                return candidate
        raise RuntimeError(
            f"could not place micromobility agent in component {spec.component_id}"
        )

    def _sample_target(self, spec: MicromobilityAgentSpec, start: np.ndarray) -> np.ndarray:
        safe_labels = self._center_admission_labels[spec.asset_id]
        start_pixel = self.regions.world_to_pixel(start)
        if start_pixel is None or int(safe_labels[start_pixel]) == 0:
            raise RuntimeError(
                f"current position for {spec.agent_id} is outside its footprint-safe raster"
            )
        safe_component = int(safe_labels[start_pixel])
        pixels = np.argwhere(safe_labels == safe_component)
        for _ in range(512):
            row, column = pixels[int(self.rng.integers(0, len(pixels)))]
            target = self.regions.pixel_to_world(int(row), int(column))
            distance = float(np.linalg.norm(target - start))
            if distance >= self.minimum_trip_m and (
                self.maximum_trip_m is None or distance <= self.maximum_trip_m
            ):
                return target
        raise RuntimeError(
            f"could not sample micromobility target in component {spec.component_id}"
        )

    def _nearest_safe_planning_start(
        self,
        spec: MicromobilityAgentSpec,
        actual_start_xy: np.ndarray,
        *,
        maximum_projection_m: float = 2.0,
    ) -> np.ndarray:
        """Return an admitted A* start near the last physically accepted pose.

        The exact OBB guard freezes every rejected vehicle candidate, but a
        legal pose can still place its centre one raster cell outside the
        conservative all-yaw erosion.  Task completion previously treated that
        harmless quantization mismatch as fatal.  Project only the *planning*
        start to the nearest safe cell in the same semantic component; the
        vehicle state is never teleported and pure pursuit rejoins the path
        from its actual pose.
        """

        actual = np.asarray(actual_start_xy, dtype=np.float64)
        safe_labels = self._center_admission_labels[spec.asset_id]
        pixel = self.regions.world_to_pixel(actual)
        if pixel is not None and int(safe_labels[pixel]) != 0:
            return actual
        pixels = self._center_admission_pixels.get((spec.asset_id, spec.component_id))
        if pixels is None or not len(pixels):
            raise RuntimeError(
                f"component {spec.component_id} has no footprint-safe centres "
                f"for {spec.asset_id}"
            )
        candidates = np.asarray(
            [
                self.regions.pixel_to_world(int(row), int(column))
                for row, column in pixels
            ],
            dtype=np.float64,
        )
        distances = np.linalg.norm(candidates - actual, axis=1)
        nearest_index = int(np.argmin(distances))
        distance = float(distances[nearest_index])
        if distance > float(maximum_projection_m):
            raise RuntimeError(
                f"current position for {spec.agent_id} is {distance:.3f} m outside "
                "its footprint-safe raster"
            )
        self.planning_start_projections[spec.agent_id] += 1
        self.maximum_planning_start_projection_m[spec.agent_id] = max(
            self.maximum_planning_start_projection_m[spec.agent_id], distance
        )
        return candidates[nearest_index]

    def _new_task(
        self,
        spec: MicromobilityAgentSpec,
        state: MicromobilityState,
        *,
        preferred_heading_rad: float | None = None,
    ) -> MicromobilityTask:
        start = self._nearest_safe_planning_start(spec, state.position_xy)
        for _ in range(1024):
            target = self._sample_target(spec, start)
            path = self._plan_path(spec, start, target)
            heading_ok = (
                preferred_heading_rad is None
                or abs(
                    wrap_angle(
                        self._path_heading(path, at_start=True)
                        - preferred_heading_rad
                    )
                )
                <= math.radians(75.0)
            )
            if (
                heading_ok
                and self._task_length_safe(path)
                and self._task_endpoints_safe(spec, path)
            ):
                return MicromobilityTask(path, target)
        raise RuntimeError(f"could not sample next footprint-safe task for {spec.agent_id}")

    @staticmethod
    def _circle_speed_cap(
        state: MicromobilityState,
        obstacle_positions_xy: np.ndarray,
        requested: float,
        *,
        stop_distance_m: float,
        slow_distance_m: float,
        lateral_threshold_m: float = 1.15,
    ) -> float:
        if len(obstacle_positions_xy) == 0:
            return requested
        forward = np.asarray((math.cos(state.yaw_rad), math.sin(state.yaw_rad)))
        relative = obstacle_positions_xy - state.position_xy
        longitudinal = relative @ forward
        lateral = np.abs(relative[:, 0] * forward[1] - relative[:, 1] * forward[0])
        candidates = longitudinal[
            (longitudinal > 0) & (lateral < float(lateral_threshold_m))
        ]
        if len(candidates) == 0:
            return requested
        distance = float(np.min(candidates))
        if distance <= stop_distance_m:
            return 0.0
        if distance >= slow_distance_m:
            return requested
        return requested * (distance - stop_distance_m) / (slow_distance_m - stop_distance_m)

    @staticmethod
    def _minimum_pedestrian_clearance_m(
        state: MicromobilityState,
        calibration: AssetCalibration,
        pedestrian_xy: np.ndarray,
        *,
        pedestrian_radius_m: float = 0.30,
        vehicle_margin_m: float = 0.10,
    ) -> float:
        """Signed circle-to-vehicle-OBB clearance for the closest person."""

        if len(pedestrian_xy) == 0:
            return float("inf")
        relative = pedestrian_xy - state.position_xy
        cosine, sine = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
        local_x = cosine * relative[:, 0] + sine * relative[:, 1]
        local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
        half_length = calibration.length_m * 0.5 + vehicle_margin_m
        half_width = calibration.width_m * 0.5 + vehicle_margin_m
        outside_x = np.maximum(np.abs(local_x) - half_length, 0.0)
        outside_y = np.maximum(np.abs(local_y) - half_width, 0.0)
        outside = np.hypot(outside_x, outside_y)
        inside = (outside_x == 0.0) & (outside_y == 0.0)
        signed_point = outside
        signed_point[inside] = -np.minimum(
            half_length - np.abs(local_x[inside]),
            half_width - np.abs(local_y[inside]),
        )
        return float(np.min(signed_point - pedestrian_radius_m))

    def mixed_states(self) -> tuple[object, ...]:
        """Expose calibrated enclosing circles to the shared avoidance pool."""

        from ..core.mixed_avoidance import MixedAgentState

        return tuple(
            MixedAgentState(
                agent_id=spec.agent_id,
                kind="micromobility",
                position_xy=self.states[spec.agent_id].position_xy,
                velocity_xy=self.states[spec.agent_id].velocity_xy,
                radius_m=math.hypot(
                    self.catalog[spec.asset_id].length_m * 0.5,
                    self.catalog[spec.asset_id].width_m * 0.5,
                ),
            )
            for spec in self.specs
        )

    def update(
        self,
        dt_s: float,
        pedestrian_positions_xyz: np.ndarray,
        *,
        shared_speed_scales: Mapping[str, float] | None = None,
        traffic_agents: Sequence[Mapping[str, object]] = (),
        go2_xy: np.ndarray | None = None,
    ) -> dict[str, MicromobilityState]:
        next_states: dict[str, MicromobilityState] = {}
        pedestrian_xy = np.asarray(pedestrian_positions_xyz, dtype=np.float64)[:, :2]
        external_circles: list[tuple[np.ndarray, float]] = [
            (
                np.asarray(item["position"], dtype=np.float64),
                float(item.get("radius", 1.5)),
            )
            for item in traffic_agents
            if item.get("status") in ("moving", "stopped")
        ]
        if go2_xy is not None:
            external_circles.append((np.asarray(go2_xy, dtype=np.float64), 0.67))
        for spec in self.specs:
            state = self.states[spec.agent_id]
            task = self.tasks[spec.agent_id]
            if spec.agent_id in self.closed_routes:
                # Keep the existing steering/avoidance below, supply a periodic
                # local target instead of sampling a new goal at the seam.
                lookahead=self._closed_lookahead(spec,state)
                task=MicromobilityTask(np.array([state.position_xy,lookahead]),lookahead+1000.)
            distances = np.linalg.norm(task.path_xy - state.position_xy, axis=1)
            nearest = int(np.argmin(distances))
            task.waypoint_index = max(task.waypoint_index, nearest + 1)
            while task.waypoint_index < len(task.path_xy) - 1 and np.linalg.norm(
                task.path_xy[task.waypoint_index] - state.position_xy
            ) < 2.0:
                task.waypoint_index += 1
            if np.linalg.norm(task.target_xy - state.position_xy) <= 1.0:
                self.completed_tasks[spec.agent_id] += 1
                try:
                    task = self._new_task(
                        spec,
                        state,
                        preferred_heading_rad=state.yaw_rad,
                    )
                except RuntimeError:
                    # Some bounded islands require a wide turnaround near an
                    # endpoint.  Prefer forward-compatible tasks, but retain a
                    # deterministic unrestricted fallback instead of leaving
                    # an agent permanently idle at the boundary.
                    task = self._new_task(spec, state)
                self.tasks[spec.agent_id] = task
            lookahead = task.path_xy[min(task.waypoint_index, len(task.path_xy) - 1)]
            local = lookahead - state.position_xy
            target_yaw = math.atan2(float(local[1]), float(local[0]))
            heading_error = wrap_angle(target_yaw - state.yaw_rad)
            calibration = self.catalog[spec.asset_id]
            lookahead_distance = max(1.2, float(np.linalg.norm(local)))
            desired_steering = math.atan2(
                2.0 * calibration.wheelbase_m * math.sin(heading_error),
                lookahead_distance,
            )
            desired_speed = spec.desired_speed_mps * max(0.30, 1.0 - abs(heading_error) / math.pi)
            shared_scale = float(
                (shared_speed_scales or {}).get(spec.agent_id, 1.0)
            )
            if shared_scale < 0.999:
                self.shared_avoidance_limited_steps[spec.agent_id] += 1
            desired_speed *= shared_scale
            pedestrian_stop_distance = (
                calibration.length_m * 0.5 + 0.30 + 0.35
            )
            desired_speed = self._circle_speed_cap(
                state,
                pedestrian_xy,
                desired_speed,
                stop_distance_m=pedestrian_stop_distance,
                slow_distance_m=pedestrian_stop_distance + 3.0,
                lateral_threshold_m=calibration.width_m * 0.5 + 0.65,
            )
            other_xy = np.asarray(
                [other.position_xy for agent_id, other in self.states.items() if agent_id != spec.agent_id]
            )
            desired_speed = self._circle_speed_cap(
                state, other_xy, desired_speed, stop_distance_m=1.8, slow_distance_m=5.0
            )
            for external_xy, external_radius in external_circles:
                desired_speed = self._circle_speed_cap(
                    state,
                    np.asarray([external_xy], dtype=np.float64),
                    desired_speed,
                    stop_distance_m=(
                        calibration.length_m * 0.5 + external_radius + 0.25
                    ),
                    slow_distance_m=(
                        calibration.length_m * 0.5 + external_radius + 2.5
                    ),
                    lateral_threshold_m=(
                        calibration.width_m * 0.5 + external_radius + 0.25
                    ),
                )
            candidate = step_bicycle(
                state,
                desired_speed_mps=desired_speed,
                desired_steering_rad=desired_steering,
                dt_s=dt_s,
                calibration=calibration,
                limits=MotionLimits(
                    maximum_speed_mps=max(0.1, spec.desired_speed_mps),
                    maximum_acceleration_mps2=0.7,
                    maximum_braking_mps2=2.5,
                    maximum_steering_rate_radps=0.45,
                ),
            )
            corners = footprint_corners(candidate, calibration, margin_m=0.15)
            collision = not self.regions.contains_points(corners, spec.component_id)
            if self.closed_paths_xy:
                x=np.linspace(-calibration.length_m/2-.15,calibration.length_m/2+.15,int(np.ceil((calibration.length_m+.3)/.1))+1)
                y=np.linspace(-calibration.width_m/2-.15,calibration.width_m/2+.15,int(np.ceil((calibration.width_m+.3)/.1))+1)
                xx,yy=np.meshgrid(x,y);co,si=math.cos(candidate.yaw_rad),math.sin(candidate.yaw_rad)
                body=candidate.position_xy+np.c_[co*xx.ravel()-si*yy.ravel(),si*xx.ravel()+co*yy.ravel()]
                collision |= not self.regions.contains_points(body,spec.component_id)
            # The longitudinal controller handles ordinary yielding.  This is
            # the final non-holonomic safety layer: it may only reject the
            # candidate (therefore brake), never translate the vehicle
            # sideways.  A centre-distance guard also covers a pedestrian
            # crossing the vehicle side, which a forward-cone test alone can
            # miss at an intersection.
            if self._minimum_pedestrian_clearance_m(
                candidate, calibration, pedestrian_xy
            ) < 0.15:
                collision = True
            candidate_radius = math.hypot(
                calibration.length_m * 0.5, calibration.width_m * 0.5
            )
            if any(
                float(np.linalg.norm(candidate.position_xy - external_xy))
                < candidate_radius + external_radius + 0.15
                for external_xy, external_radius in external_circles
            ):
                collision = True
                self.external_emergency_stop_steps[spec.agent_id] += 1
            for other_id, other in {**self.states, **next_states}.items():
                if other_id == spec.agent_id:
                    continue
                other_spec = next(item for item in self.specs if item.agent_id == other_id)
                other_corners = footprint_corners(other, self.catalog[other_spec.asset_id], margin_m=0.15)
                if polygons_intersect(corners, other_corners):
                    collision = True
                    break
            if collision:
                self.emergency_stops[spec.agent_id] += 1
                self.consecutive_emergency_stop_s[spec.agent_id] += float(dt_s)
                # Do not keep the translation from a rejected candidate.  A
                # braking integration still advances the body for several
                # frames and used to leak an already-rejected footprint across
                # the boundary.  Freeze pose, monotonically reduce the scalar
                # speed, and retry from the last admitted pose next frame.
                candidate = replace(
                    state,
                    speed_mps=max(0.0, state.speed_mps - 2.5 * dt_s),
                )
                # Head-on encounters in a narrow mixed-use block can leave two
                # non-holonomic agents yielding forever.  Recovery changes only
                # the goal/path after a sustained stop; it never teleports,
                # reverses, translates laterally, or bypasses the OBB guard.
                if (
                    self.consecutive_emergency_stop_s[spec.agent_id]
                    >= self.stall_replan_after_s
                    and not self.closed_paths_xy
                ):
                    try:
                        self.tasks[spec.agent_id] = self._new_task(
                            spec,
                            state,
                            preferred_heading_rad=state.yaw_rad,
                        )
                        self.stall_replans[spec.agent_id] += 1
                    except (ValueError, RuntimeError):
                        # Keep safely waiting when the current island has no
                        # forward-compatible alternative target.  The next
                        # retry happens only after another full timeout.
                        pass
                    finally:
                        self.consecutive_emergency_stop_s[spec.agent_id] = 0.0
            else:
                self.consecutive_emergency_stop_s[spec.agent_id] = 0.0
            next_states[spec.agent_id] = candidate
        self.states = next_states
        return dict(self.states)

    def metrics(self) -> dict[str, object]:
        return {
            "agent_count": len(self.specs),
            "completed_tasks": self.completed_tasks,
            "emergency_stop_steps": self.emergency_stops,
            "stall_replans": self.stall_replans,
            "shared_avoidance_limited_steps": self.shared_avoidance_limited_steps,
            "external_emergency_stop_steps": self.external_emergency_stop_steps,
            "planning_start_projections": self.planning_start_projections,
            "maximum_planning_start_projection_m": (
                self.maximum_planning_start_projection_m
            ),
            "stall_replan_after_s": self.stall_replan_after_s,
            "minimum_trip_m": self.minimum_trip_m,
            "maximum_trip_m": self.maximum_trip_m,
            "controller": "A* block task + pure-pursuit bicycle kinematics",
            "avoidance": "shared reciprocal mixed-agent speed limits + pedestrian/external longitudinal braking + exact footprint admission",
        }


def attach_official_people_dynamic_obstacle(prim: object) -> str:
    """Attach NVIDIA's installed DynamicObstacle BehaviorScript to one prim.

    Mirrors ``omni.anim.people.python_ext.add_dynamic_obstacle_behavior_script``:
    apply the OmniScriptingAPI schema via the USD command, then write the raw
    ``omni:scripting:scripts`` attribute (Isaac 4.5's pxr no longer generates
    ``CreateScriptsAttr`` on ``OmniScriptingAPI``).
    """

    import inspect

    import omni.kit.commands
    from pxr import Sdf
    from omni.anim.people.scripts import dynamic_obstacle

    script_path = Path(inspect.getfile(dynamic_obstacle)).resolve()
    prim_path = str(prim.GetPath())
    omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim_path)])
    attr = prim.GetAttribute("omni:scripting:scripts")
    script_list = [str(script_path)]
    existing = attr.Get()
    if existing:
        script_list.extend(str(value) for value in existing)
    attr.Set(script_list)
    return str(script_path)
