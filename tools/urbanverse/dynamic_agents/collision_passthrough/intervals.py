"""Geometric crossing state and navigation-training validity annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass
class PassthroughIntervalTracker:
    center_xy: tuple[float, float]
    dimensions_xy: tuple[float, float]
    yaw_deg: float
    route_direction_xy: tuple[float, float]
    robot_proxy_radius_m: float
    mode: str
    _rows: list[dict[str, Any]] = field(default_factory=list)
    _phase_started_step: int | None = None
    _phase_started_s: float | None = None
    _last_phase: str | None = None
    _intervals: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in {"baseline", "passthrough"}:
            raise ValueError("mode must be baseline or passthrough")
        direction = np.asarray(self.route_direction_xy, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            raise ValueError("route direction must be non-zero")
        self._route_direction = direction / norm
        angle = math.radians(float(self.yaw_deg))
        self._world_to_local = np.asarray(
            [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]],
            dtype=np.float64,
        )

    def _state(self, position_xy: np.ndarray) -> tuple[str, float, bool]:
        local = self._world_to_local @ (position_xy - np.asarray(self.center_xy))
        # Route progress through the obstacle is measured along the actual route,
        # while overlap uses the configured obstacle-oriented rectangle.
        signed_progress = float(np.dot(position_xy - np.asarray(self.center_xy), self._route_direction))
        projected_half_extent = 0.5 * (
            abs(float(np.dot(self._route_direction, self._world_to_local[0]))) * self.dimensions_xy[0]
            + abs(float(np.dot(self._route_direction, self._world_to_local[1]))) * self.dimensions_xy[1]
        )
        proxy_overlap = bool(
            abs(local[0]) <= 0.5 * self.dimensions_xy[0] + self.robot_proxy_radius_m
            and abs(local[1]) <= 0.5 * self.dimensions_xy[1] + self.robot_proxy_radius_m
        )
        core = bool(
            abs(local[0]) <= 0.5 * self.dimensions_xy[0]
            and abs(local[1]) <= 0.5 * self.dimensions_xy[1]
        )
        if signed_progress < -projected_half_extent - self.robot_proxy_radius_m:
            phase = "approach"
        elif signed_progress > projected_half_extent + self.robot_proxy_radius_m:
            phase = "exited"
        elif core:
            phase = "inside_obstacle_core"
        elif proxy_overlap:
            phase = "intersecting_obstacle_proxy"
        else:
            phase = "beside_obstacle"
        return phase, signed_progress, proxy_overlap

    def update(self, step_index: int, timestamp_s: float, position_xy: list[float] | np.ndarray) -> dict[str, Any]:
        position = np.asarray(position_xy, dtype=np.float64)
        phase, signed_progress, proxy_overlap = self._state(position)
        if self._last_phase is None:
            self._last_phase = phase
            self._phase_started_step = int(step_index)
            self._phase_started_s = float(timestamp_s)
        elif phase != self._last_phase:
            self._intervals.append(
                {
                    "phase": self._last_phase,
                    "start_step_index": self._phase_started_step,
                    "end_step_index_inclusive": int(step_index) - 1,
                    "start_s": self._phase_started_s,
                    "end_s_exclusive": float(timestamp_s),
                }
            )
            self._last_phase = phase
            self._phase_started_step = int(step_index)
            self._phase_started_s = float(timestamp_s)
        invalid = bool(self.mode == "passthrough" and phase in {
            "intersecting_obstacle_proxy",
            "inside_obstacle_core",
        })
        row = {
            "obstacle_phase": phase,
            "signed_progress_through_obstacle_m": signed_progress,
            "robot_proxy_overlaps_obstacle": proxy_overlap,
            "passthrough_physics_active": self.mode == "passthrough",
            "valid_for_navigation_training": not invalid,
            "navigation_training_invalid_reason": (
                "selected_obstacle_collision_response_disabled"
                if invalid
                else None
            ),
        }
        self._rows.append({"step_index": int(step_index), "timestamp_s": float(timestamp_s), **row})
        return row

    def finalize(self, final_timestamp_s: float) -> dict[str, Any]:
        if self._last_phase is not None:
            last_step = self._rows[-1]["step_index"] if self._rows else 0
            self._intervals.append(
                {
                    "phase": self._last_phase,
                    "start_step_index": self._phase_started_step,
                    "end_step_index_inclusive": last_step,
                    "start_s": self._phase_started_s,
                    "end_s_exclusive": float(final_timestamp_s),
                }
            )
            self._last_phase = None
        phases = {row["obstacle_phase"] for row in self._rows}
        invalid_rows = [row for row in self._rows if not row["valid_for_navigation_training"]]
        invalid_intervals = [
            row for row in self._intervals
            if self.mode == "passthrough" and row["phase"] in {
                "intersecting_obstacle_proxy", "inside_obstacle_core"
            }
        ]
        return {
            "mode": self.mode,
            "entered_obstacle_proxy": "intersecting_obstacle_proxy" in phases,
            "entered_obstacle_core": "inside_obstacle_core" in phases,
            "exited_obstacle": "exited" in phases,
            "crossed_obstacle": "inside_obstacle_core" in phases and "exited" in phases,
            "phase_intervals": self._intervals,
            "invalid_intervals": invalid_intervals,
            "invalid_step_count": len(invalid_rows),
            "invalid_step_indices": [row["step_index"] for row in invalid_rows],
        }

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)


class MultiObstacleIntervalTracker:
    """Union validity plus per-obstacle enter/core/exit evidence."""

    def __init__(self, obstacles: list[Any], route_direction_xy: tuple[float, float], mode: str):
        self.mode = mode
        self._trackers = {
            obstacle.obstacle_id: PassthroughIntervalTracker(
                center_xy=obstacle.center_xyz[:2],
                dimensions_xy=obstacle.dimensions_xyz[:2],
                yaw_deg=obstacle.yaw_deg,
                route_direction_xy=route_direction_xy,
                robot_proxy_radius_m=obstacle.robot_proxy_radius_m,
                mode=mode,
            )
            for obstacle in obstacles
        }
        self._rows: list[dict[str, Any]] = []

    def update(
        self, step_index: int, timestamp_s: float, position_xy: list[float] | np.ndarray
    ) -> dict[str, Any]:
        states = {
            obstacle_id: tracker.update(step_index, timestamp_s, position_xy)
            for obstacle_id, tracker in self._trackers.items()
        }
        overlaps = [
            obstacle_id
            for obstacle_id, state in states.items()
            if state["robot_proxy_overlaps_obstacle"]
        ]
        cores = [
            obstacle_id
            for obstacle_id, state in states.items()
            if state["obstacle_phase"] == "inside_obstacle_core"
        ]
        if cores:
            phase = "inside_" + cores[0]
        elif overlaps:
            phase = "overlap_" + overlaps[0]
        elif all(state["obstacle_phase"] == "exited" for state in states.values()):
            phase = "all_obstacles_exited"
        elif any(state["obstacle_phase"] == "exited" for state in states.values()):
            phase = "between_obstacles"
        else:
            phase = "approach"
        invalid = bool(self.mode == "passthrough" and overlaps)
        row = {
            "obstacle_phase": phase,
            "overlapping_obstacle_ids": overlaps,
            "inside_obstacle_core_ids": cores,
            "obstacle_states": states,
            "passthrough_physics_active": self.mode == "passthrough",
            "valid_for_navigation_training": not invalid,
            "navigation_training_invalid_reason": (
                "default_world_obstacle_collision_response_disabled" if invalid else None
            ),
        }
        self._rows.append({"step_index": step_index, "timestamp_s": timestamp_s, **row})
        return row

    def finalize(self, final_timestamp_s: float) -> dict[str, Any]:
        obstacle_summaries = {
            obstacle_id: tracker.finalize(final_timestamp_s)
            for obstacle_id, tracker in self._trackers.items()
        }
        invalid_rows = [row for row in self._rows if not row["valid_for_navigation_training"]]
        invalid_intervals = []
        start = None
        last = None
        active_ids: tuple[str, ...] = ()
        for row in self._rows:
            row_ids = tuple(row["overlapping_obstacle_ids"])
            if not row["valid_for_navigation_training"]:
                if start is None or row_ids != active_ids:
                    if start is not None and last is not None:
                        invalid_intervals.append(
                            {
                                "start_step_index": start["step_index"],
                                "end_step_index_inclusive": last["step_index"],
                                "start_s": start["timestamp_s"],
                                "end_s_exclusive": row["timestamp_s"],
                                "obstacle_ids": list(active_ids),
                            }
                        )
                    start = row
                    active_ids = row_ids
                last = row
            elif start is not None and last is not None:
                invalid_intervals.append(
                    {
                        "start_step_index": start["step_index"],
                        "end_step_index_inclusive": last["step_index"],
                        "start_s": start["timestamp_s"],
                        "end_s_exclusive": row["timestamp_s"],
                        "obstacle_ids": list(active_ids),
                    }
                )
                start = last = None
                active_ids = ()
        if start is not None and last is not None:
            invalid_intervals.append(
                {
                    "start_step_index": start["step_index"],
                    "end_step_index_inclusive": last["step_index"],
                    "start_s": start["timestamp_s"],
                    "end_s_exclusive": final_timestamp_s,
                    "obstacle_ids": list(active_ids),
                }
            )
        return {
            "mode": self.mode,
            "obstacles": obstacle_summaries,
            "all_obstacles_crossed": all(
                summary["crossed_obstacle"] for summary in obstacle_summaries.values()
            ),
            "crossed_obstacle_ids": [
                obstacle_id
                for obstacle_id, summary in obstacle_summaries.items()
                if summary["crossed_obstacle"]
            ],
            "invalid_intervals": invalid_intervals,
            "invalid_step_count": len(invalid_rows),
            "invalid_step_indices": [row["step_index"] for row in invalid_rows],
        }
