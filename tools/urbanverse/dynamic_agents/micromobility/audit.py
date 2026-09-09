"""Scene-independent admission and interaction audit for mixed walkable agents."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..pedestrians.roaming import RoamingAssignment
from ..pedestrians.walkable_regions import WalkableRegions
from .avoidance import polygons_intersect
from .calibration import AssetCalibration
from .motion import MicromobilityState, footprint_corners
from .roaming import MicromobilityAgentSpec


def pedestrian_to_vehicle_clearance_m(
    pedestrian_xy: np.ndarray,
    state: MicromobilityState,
    calibration: AssetCalibration,
    *,
    pedestrian_radius_m: float = 0.30,
    vehicle_margin_m: float = 0.05,
) -> float:
    """Signed circle-to-oriented-rectangle clearance in metres."""

    relative = np.asarray(pedestrian_xy, dtype=np.float64) - state.position_xy
    cosine, sine = np.cos(state.yaw_rad), np.sin(state.yaw_rad)
    local = np.asarray(
        (
            cosine * relative[0] + sine * relative[1],
            -sine * relative[0] + cosine * relative[1],
        ),
        dtype=np.float64,
    )
    half = np.asarray(
        (
            calibration.length_m * 0.5 + vehicle_margin_m,
            calibration.width_m * 0.5 + vehicle_margin_m,
        ),
        dtype=np.float64,
    )
    outside = np.maximum(np.abs(local) - half, 0.0)
    signed_point_distance = (
        float(np.linalg.norm(outside))
        if np.any(outside > 0.0)
        else -float(np.min(half - np.abs(local)))
    )
    return signed_point_distance - float(pedestrian_radius_m)


class MixedWalkableAgentAudit:
    """Accumulate portable safety evidence for People and two-wheelers.

    The audit consumes only approved-region geometry and controller states. It
    therefore works unchanged for every scene that has passed the walkable-
    region admission gate; no Scene09 coordinates or USD paths are embedded.
    """

    def __init__(
        self,
        regions: WalkableRegions,
        people: Sequence[RoamingAssignment],
        micromobility: Sequence[MicromobilityAgentSpec],
        catalog: Mapping[str, AssetCalibration],
        *,
        micromobility_regions: WalkableRegions | None = None,
        approved_union_regions: WalkableRegions | None = None,
        people_raster_tolerance_m: float = 0.35,
        footprint_margin_m: float = 0.15,
        people_radius_m: float = 0.30,
        people_visual_penetration_tolerance_m: float = 0.10,
    ) -> None:
        self.regions = regions
        self.micromobility_regions = micromobility_regions or regions
        self.approved_union_regions = approved_union_regions or regions
        self.people = tuple(people)
        self.micromobility = tuple(micromobility)
        self.catalog = dict(catalog)
        self.people_raster_tolerance_m = float(people_raster_tolerance_m)
        self.footprint_margin_m = float(footprint_margin_m)
        self.people_radius_m = float(people_radius_m)
        self.people_visual_penetration_tolerance_m = float(
            people_visual_penetration_tolerance_m
        )
        self.people_component_violations = 0
        self.people_approved_union_violations = 0
        self.people_raster_boundary_excursions = 0
        self.micromobility_component_violations = 0
        self.micromobility_obb_overlap_events = 0
        self.people_people_overlap_events = 0
        self.people_people_severe_overlap_events = 0
        self.minimum_people_people_clearance_m = float("inf")
        self.closest_people_interaction: dict[str, object] | None = None
        self.minimum_people_micromobility_clearance_m = float("inf")
        self.close_interaction_samples = 0
        self.closest_interaction: dict[str, object] | None = None
        self.sample_count = 0
        self.initial_people_xy: np.ndarray | None = None
        self.initial_micromobility_xy: np.ndarray | None = None
        self.maximum_people_displacement_m = 0.0
        self.maximum_micromobility_displacement_m = 0.0

    def observe(
        self,
        people_positions_xyz: np.ndarray,
        micromobility_states: Mapping[str, MicromobilityState],
        *,
        simulation_time_s: float,
    ) -> None:
        people_positions = np.asarray(people_positions_xyz, dtype=np.float64)
        if people_positions.shape != (len(self.people), 3):
            raise ValueError("People position count does not match roaming assignments")
        people_xy = people_positions[:, :2]
        micro_xy = np.asarray(
            [micromobility_states[item.agent_id].position_xy for item in self.micromobility],
            dtype=np.float64,
        )
        if self.initial_people_xy is None:
            self.initial_people_xy = people_xy.copy()
            self.initial_micromobility_xy = micro_xy.copy()
        self.sample_count += 1
        self.maximum_people_displacement_m = max(
            self.maximum_people_displacement_m,
            float(np.linalg.norm(people_xy - self.initial_people_xy, axis=1).max()),
        )
        if len(micro_xy):
            self.maximum_micromobility_displacement_m = max(
                self.maximum_micromobility_displacement_m,
                float(
                    np.linalg.norm(
                        micro_xy - self.initial_micromobility_xy, axis=1
                    ).max()
                ),
            )

        for assignment, position in zip(self.people, people_xy):
            self.people_approved_union_violations += int(
                not self.approved_union_regions.is_admitted_with_raster_tolerance(
                    position, self.people_raster_tolerance_m
                )
            )
            exact_component = self.regions.component_at(position)
            if exact_component != assignment.component_id:
                self.people_raster_boundary_excursions += 1
                self.people_component_violations += int(
                    not self.regions.in_component_with_raster_tolerance(
                        position,
                        assignment.component_id,
                        tolerance_m=self.people_raster_tolerance_m,
                    )
                )

        if len(people_xy) > 1:
            pairwise = np.linalg.norm(
                people_xy[:, None, :] - people_xy[None, :, :], axis=2
            )
            upper = pairwise[np.triu_indices(len(people_xy), k=1)]
            signed = upper - 2.0 * self.people_radius_m
            minimum_index = int(np.argmin(signed))
            minimum_clearance = float(signed[minimum_index])
            if minimum_clearance < self.minimum_people_people_clearance_m:
                upper_indices = np.triu_indices(len(people_xy), k=1)
                first = int(upper_indices[0][minimum_index])
                second = int(upper_indices[1][minimum_index])
                self.minimum_people_people_clearance_m = minimum_clearance
                self.closest_people_interaction = {
                    "simulation_time_s": float(simulation_time_s),
                    "first_person_index": first,
                    "second_person_index": second,
                    "first_person_xy": people_xy[first].tolist(),
                    "second_person_xy": people_xy[second].tolist(),
                    "signed_clearance_m": minimum_clearance,
                }
            self.people_people_overlap_events += int(np.count_nonzero(signed < 0.0))
            self.people_people_severe_overlap_events += int(
                np.count_nonzero(
                    signed < -self.people_visual_penetration_tolerance_m
                )
            )

        footprints: dict[str, np.ndarray] = {}
        for spec in self.micromobility:
            footprint = footprint_corners(
                micromobility_states[spec.agent_id],
                self.catalog[spec.asset_id],
                margin_m=self.footprint_margin_m,
            )
            footprints[spec.agent_id] = footprint
            self.micromobility_component_violations += int(
                not self.micromobility_regions.contains_points(
                    footprint, spec.component_id
                )
            )
        ids = tuple(footprints)
        for first_index, first_id in enumerate(ids):
            for second_id in ids[first_index + 1 :]:
                self.micromobility_obb_overlap_events += int(
                    polygons_intersect(footprints[first_id], footprints[second_id])
                )

        if not len(micro_xy):
            return
        centre_distances = np.linalg.norm(
            people_xy[:, None, :] - micro_xy[None, :, :], axis=2
        )
        self.close_interaction_samples += int(float(centre_distances.min()) <= 4.0)
        for person_index, person_xy in enumerate(people_xy):
            for spec in self.micromobility:
                state = micromobility_states[spec.agent_id]
                clearance = pedestrian_to_vehicle_clearance_m(
                    person_xy, state, self.catalog[spec.asset_id]
                )
                if clearance < self.minimum_people_micromobility_clearance_m:
                    self.minimum_people_micromobility_clearance_m = clearance
                    self.closest_interaction = {
                        "simulation_time_s": float(simulation_time_s),
                        "person_index": person_index,
                        "person_xy": person_xy.tolist(),
                        "micromobility_id": spec.agent_id,
                        "micromobility_xy": state.position_xy.tolist(),
                        "micromobility_speed_mps": float(state.speed_mps),
                        "signed_clearance_m": float(clearance),
                    }

    def summary(self) -> dict[str, object]:
        minimum_clearance = self.minimum_people_micromobility_clearance_m
        people_clearance = self.minimum_people_people_clearance_m
        return {
            "sample_count": self.sample_count,
            "people_component_violations": self.people_component_violations,
            "people_approved_union_violations": (
                self.people_approved_union_violations
            ),
            "people_raster_boundary_excursions": self.people_raster_boundary_excursions,
            "people_raster_membership_tolerance_m": self.people_raster_tolerance_m,
            "micromobility_component_violations": (
                self.micromobility_component_violations
            ),
            "micromobility_obb_overlap_events": self.micromobility_obb_overlap_events,
            "people_people_overlap_events": self.people_people_overlap_events,
            "people_people_severe_overlap_events": (
                self.people_people_severe_overlap_events
            ),
            "people_visual_penetration_tolerance_m": (
                self.people_visual_penetration_tolerance_m
            ),
            "minimum_people_people_clearance_m": (
                None if people_clearance == float("inf") else people_clearance
            ),
            "closest_people_interaction": self.closest_people_interaction,
            "people_collision_radius_m": self.people_radius_m,
            "minimum_people_micromobility_clearance_m": (
                None if minimum_clearance == float("inf") else minimum_clearance
            ),
            "close_interaction_samples": self.close_interaction_samples,
            "closest_interaction": self.closest_interaction,
            "maximum_people_displacement_m": self.maximum_people_displacement_m,
            "maximum_micromobility_displacement_m": (
                self.maximum_micromobility_displacement_m
            ),
        }
