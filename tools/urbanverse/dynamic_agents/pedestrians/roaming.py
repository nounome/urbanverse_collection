"""Fixed-population official People roaming within approved connected blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
from typing import Mapping, Sequence

import numpy as np

from .official_runtime import OfficialCharacterSpec, OfficialPeopleRuntime
from .walkable_regions import WalkableRegions


@dataclass(frozen=True)
class RoamingAssignment:
    name: str
    asset: Path
    component_id: int
    start_xyz: tuple[float, float, float]
    initial_target_xyz: tuple[float, float, float]


def _separated_sample(
    regions: WalkableRegions,
    component_id: int,
    rng: np.random.Generator,
    existing_xy: Sequence[np.ndarray],
    minimum_distance_m: float,
    attempts: int = 512,
) -> np.ndarray:
    for _ in range(attempts):
        candidate = regions.sample(component_id, rng)
        if all(np.linalg.norm(candidate - point) >= minimum_distance_m for point in existing_xy):
            return candidate
    raise RuntimeError(
        f"could not sample component {component_id} with {minimum_distance_m:.2f} m separation"
    )


def _bounded_separated_sample(
    regions: WalkableRegions,
    component_id: int,
    rng: np.random.Generator,
    origin_xy: np.ndarray,
    minimum_distance_m: float,
    maximum_distance_m: float,
    attempts: int = 1024,
) -> np.ndarray:
    """Sample inside one connected block within a radial envelope of origin.

    Recast's ``query_shortest_path`` can enter a GIL-holding infinite search for
    certain long-distance pairs on the Scene09 helper (empirically: probes that
    stayed local all resolved, while far-apart pairs spun at 650% CPU with no
    Python-interruptible window).  Keeping every GoTo segment inside a modest
    radius mirrors how NVIDIA's own people waypoints are laid out and sidesteps
    the pathological search.
    """
    for _ in range(attempts):
        candidate = regions.sample(component_id, rng)
        distance = float(np.linalg.norm(candidate - origin_xy))
        if minimum_distance_m <= distance <= maximum_distance_m:
            return candidate
    raise RuntimeError(
        f"could not sample component {component_id} "
        f"within [{minimum_distance_m:.2f}, {maximum_distance_m:.2f}] m of origin"
    )


def plan_roaming_assignments(
    regions: WalkableRegions,
    assets: Sequence[Path],
    *,
    count: int,
    seed: int,
    minimum_spawn_separation_m: float = 1.5,
    minimum_initial_trip_m: float = 6.0,
    maximum_initial_trip_m: float = 30.0,
    multi_resident_minimum_width_m: float | None = None,
) -> tuple[RoamingAssignment, ...]:
    """Allocate a deterministic resident population proportional to block area."""

    if count <= 0 or not assets:
        raise ValueError("count and assets must be nonempty")
    rng = np.random.default_rng(seed)
    component_ids = np.asarray(regions.component_ids, dtype=np.int32)
    weights = np.asarray(
        [len(regions.component_pixels[int(component)]) for component in component_ids],
        dtype=np.float64,
    )
    weights /= weights.sum()
    # Preserve area weighting without allowing one large component to absorb
    # most of a dense resident population.  The portable cap is derived only
    # from admitted component count, so it generalizes to new scenes.
    capacity = int(math.ceil(count / len(component_ids)))
    capacities = np.full(len(component_ids), capacity, dtype=np.int32)
    if multi_resident_minimum_width_m is not None:
        if multi_resident_minimum_width_m <= 0.0:
            raise ValueError("multi_resident_minimum_width_m must be positive")
        for index, component in enumerate(component_ids):
            if regions.component_maximum_width_m(int(component)) < multi_resident_minimum_width_m:
                capacities[index] = 1
            else:
                # Narrow components retain exactly one representative, while
                # genuinely wide shared blocks absorb the remaining requested
                # population.  The old uniform cap made that combination
                # impossible whenever more than ``ceil(count / blocks)``
                # residents had to move into a wide block.
                capacities[index] = count
        if int(capacities.sum()) < count:
            raise RuntimeError(
                "walkable components cannot admit the requested population "
                "under the narrow-component single-resident rule"
            )
    guarantee_component_coverage = multi_resident_minimum_width_m is not None
    if guarantee_component_coverage and count < len(component_ids):
        raise RuntimeError(
            "requested population is smaller than the number of admitted components"
        )
    # With the explicit narrow-component policy, every approved block gets one
    # resident first.  Area weighting applies only to the remainder; otherwise
    # a narrow or small block can be randomly left empty despite being admitted.
    used = (
        np.ones(len(component_ids), dtype=np.int32)
        if guarantee_component_coverage
        else np.zeros(len(component_ids), dtype=np.int32)
    )
    assigned_components: list[int] = (
        [int(value) for value in component_ids]
        if guarantee_component_coverage
        else []
    )
    for _ in range(count - len(assigned_components)):
        available = np.flatnonzero(used < capacities)
        if guarantee_component_coverage:
            # Deterministic weighted round-robin avoids a seeded random draw
            # placing almost the entire remainder in one wide block.
            selected = int(available[np.argmin(used[available] / weights[available])])
        else:
            local_weights = weights[available]
            local_weights /= local_weights.sum()
            selected = int(rng.choice(available, p=local_weights))
        used[selected] += 1
        assigned_components.append(int(component_ids[selected]))
    existing_by_component: dict[int, list[np.ndarray]] = {
        int(component): [] for component in component_ids
    }
    assignments: list[RoamingAssignment] = []
    for index, component_value in enumerate(assigned_components):
        component = int(component_value)
        start = _separated_sample(
            regions,
            component,
            rng,
            existing_by_component[component],
            minimum_spawn_separation_m,
        )
        existing_by_component[component].append(start)
        target = _bounded_separated_sample(
            regions,
            component,
            rng,
            start,
            minimum_initial_trip_m,
            maximum_initial_trip_m,
        )
        z = regions.config.ground_z_m
        assignments.append(
            RoamingAssignment(
                name=f"Resident_{index:02d}",
                asset=Path(assets[index % len(assets)]).resolve(),
                component_id=component,
                start_xyz=(float(start[0]), float(start[1]), z),
                initial_target_xyz=(float(target[0]), float(target[1]), z),
            )
        )
    return tuple(assignments)


def assignments_to_specs(
    assignments: Sequence[RoamingAssignment],
    waypoint_loops: Mapping[str, np.ndarray] | None = None,
) -> tuple[OfficialCharacterSpec, ...]:
    """Build official specs; a closed waypoint ring replaces the single target.

    When ``waypoint_loops`` provides a per-character ring of world xy points,
    the official runtime walks the whole ring as one multi-point ``GoTo`` whose
    segments were validated on the baked NavMesh during setup.  This keeps the
    runtime walking with zero in-loop Recast re-queries.
    """
    loops = waypoint_loops or {}
    return tuple(
        OfficialCharacterSpec(
            name=item.name,
            asset=item.asset,
            start_xyz=item.start_xyz,
            waypoints_xyz=tuple(
                (float(x), float(y), item.start_xyz[2])
                for x, y in loops.get(
                    item.name,
                    [(item.initial_target_xyz[0], item.initial_target_xyz[1])],
                )
            ),
        )
        for item in assignments
    )


def build_roaming_waypoint_loops(
    regions: WalkableRegions,
    assignments: Sequence[RoamingAssignment],
    rng: np.random.Generator,
    *,
    ring_points: int = 7,
    radius_min_m: float = 12.0,
    radius_max_m: float = 18.0,
    attempts: int = 4096,
    minimum_leg_m: float = 5.0,
    maximum_leg_m: float = 30.0,
    territory_margin_m: float = 0.75,
    partition_shared_components: bool = True,
) -> dict[str, np.ndarray]:
    """Build cyclic, seeded walking targets for every resident.

    Every ring point is sampled inside a modest radial band around the
    character's start within its connected walkable component, then ordered by
    angle so consecutive points (including the implicit last-to-first closing
    leg) are short.  The first point is the resident start and is deliberately
    not duplicated at the end: the manager performs modulo wraparound.  Runtime
    target selection is therefore deterministic, component-local and bounded;
    it never performs Python-side random long-distance Recast searches.
    """
    if territory_margin_m < 0.0:
        raise ValueError("territory_margin_m cannot be negative")
    starts_by_component: dict[int, list[tuple[str, np.ndarray]]] = {}
    for assignment in assignments:
        starts_by_component.setdefault(assignment.component_id, []).append(
            (
                assignment.name,
                np.asarray(assignment.start_xyz[:2], dtype=np.float64),
            )
        )

    loops: dict[str, np.ndarray] = {}
    for assignment in assignments:
        component = assignment.component_id
        start = np.asarray(assignment.start_xyz[:2], dtype=np.float64)
        neighbours = starts_by_component[component]
        samples: list[np.ndarray] = []
        radial_samples: list[np.ndarray] = []
        component_samples: list[np.ndarray] = []
        for _ in range(attempts):
            candidate = regions.sample(component, rng)
            component_samples.append(candidate)
            radius = float(np.linalg.norm(candidate - start))
            if not (radius_min_m <= radius <= radius_max_m):
                continue
            radial_samples.append(candidate)
            # More than one resident may occupy a long, narrow semantic block.
            # Independent random rings then cross even though NVIDIA's local
            # avoidance is enabled.  Restrict every resident to the Voronoi
            # territory around its seeded start, with a small neutral gap at
            # shared boundaries.  The territory is derived from the approved
            # component itself, so it remains portable and does not introduce
            # Scene09-specific coordinates or pose control.
            other_distances = [
                float(np.linalg.norm(candidate - other_start))
                for other_name, other_start in neighbours
                if other_name != assignment.name
            ]
            if (
                partition_shared_components
                and other_distances
                and radius + territory_margin_m > min(other_distances)
            ):
                continue
            samples.append(candidate)
        def _ring_from(candidates: Sequence[np.ndarray]) -> np.ndarray | None:
            if len(candidates) < 3:
                return None
            samples_array = np.asarray(candidates, dtype=np.float64)
            angles = np.arctan2(
                samples_array[:, 1] - start[1], samples_array[:, 0] - start[0]
            )
            sorted_samples = samples_array[np.argsort(angles)]
            count = min(ring_points, len(sorted_samples))
            picked = sorted_samples[
                np.linspace(0, len(sorted_samples) - 1, count).astype(np.int32)
            ]
            ring = [start]
            for point in picked:
                distance = float(np.linalg.norm(point - ring[-1]))
                if minimum_leg_m <= distance <= maximum_leg_m:
                    ring.append(point)
            while len(ring) > 2 and not (
                minimum_leg_m
                <= float(np.linalg.norm(ring[-1] - start))
                <= maximum_leg_m
            ):
                ring.pop()
            if len(ring) < 3:
                return None
            return np.asarray(ring, dtype=np.float64)

        def _shared_random_ring(candidates: Sequence[np.ndarray]) -> np.ndarray | None:
            """Build a long-leg random cycle in a shared connected block.

            Angular sorting was designed for small private rings and collapses
            to short legs when several residents share a block.  In shared
            mode, choose each next goal relative to the current goal and only
            accept the final point when it can also return to the start.  The
            resulting deterministic seeded cycle is consumed one target at a
            time by the official CharacterManager.
            """

            if not candidates:
                return None
            candidate_array = np.asarray(candidates, dtype=np.float64)
            for count in range(ring_points, 2, -1):
                for _ in range(256):
                    ring = [start]
                    used: set[int] = set()
                    for point_index in range(1, count):
                        distances = np.linalg.norm(
                            candidate_array - ring[-1], axis=1
                        )
                        eligible = np.flatnonzero(
                            (distances >= minimum_leg_m)
                            & (distances <= maximum_leg_m)
                        )
                        if point_index == count - 1:
                            closing = np.linalg.norm(
                                candidate_array - start, axis=1
                            )
                            eligible = eligible[
                                (closing[eligible] >= minimum_leg_m)
                                & (closing[eligible] <= maximum_leg_m)
                            ]
                        eligible = np.asarray(
                            [value for value in eligible if int(value) not in used],
                            dtype=np.int64,
                        )
                        if not len(eligible):
                            break
                        selected = int(rng.choice(eligible))
                        used.add(selected)
                        ring.append(candidate_array[selected])
                    if len(ring) == count:
                        return np.asarray(ring, dtype=np.float64)
            return None

        ring = (
            _ring_from(samples)
            if partition_shared_components
            else _shared_random_ring(component_samples)
        )
        # Extremely dense synthetic/admission cases can leave a Voronoi cell
        # too small for a valid cyclic ring.  Preserve the historical
        # component-local fallback there; production admission separately
        # checks the final paths and cannot silently accept a missing loop.
        if ring is None and partition_shared_components:
            ring = _ring_from(radial_samples)
        if ring is not None:
            loops[assignment.name] = ring
    return loops


def align_initial_targets_to_waypoint_loops(
    assignments: Sequence[RoamingAssignment],
    waypoint_loops: Mapping[str, np.ndarray],
) -> tuple[RoamingAssignment, ...]:
    """Make the first official GoTo leg obey the resident territory.

    Assignment targets are sampled before per-resident territories exist.  If
    those old targets are sent to CharacterManager, two residents can cross on
    their very first leg even though every later ring target is disjoint.
    Replace only that initial target with ring[1]; pose execution remains fully
    owned by the official CharacterManager/NavMesh runtime.
    """

    aligned: list[RoamingAssignment] = []
    for assignment in assignments:
        ring = np.asarray(waypoint_loops[assignment.name], dtype=np.float64)
        if ring.ndim != 2 or ring.shape[0] < 2 or ring.shape[1] != 2:
            raise ValueError(f"invalid waypoint loop for {assignment.name}")
        aligned.append(
            replace(
                assignment,
                initial_target_xyz=(
                    float(ring[1, 0]),
                    float(ring[1, 1]),
                    float(assignment.start_xyz[2]),
                ),
            )
        )
    return tuple(aligned)


def expand_waypoint_loops_with_grid_paths(
    regions: WalkableRegions,
    assignments: Sequence[RoamingAssignment],
    waypoint_loops: Mapping[str, np.ndarray],
    *,
    maximum_navigation_leg_m: float = 7.5,
) -> dict[str, np.ndarray]:
    """Expand long logical goals into short, same-component NavMesh commands.

    Isaac Sim 4.5 can hang inside ``query_shortest_path`` for a small subset of
    otherwise valid 15--25 m Scene09 queries.  Plan the long task on the
    already-approved raster, then submit consecutive short waypoints to the
    official CharacterManager.  Intermediate points follow one continuous
    path; they do not introduce random heading changes or direct pose writes.
    """

    if maximum_navigation_leg_m <= 0.0:
        raise ValueError("maximum_navigation_leg_m must be positive")
    # Local import avoids a module-import cycle: micromobility also consumes
    # RoamingAssignment for its mixed-agent audit.
    from ..micromobility.roaming import plan_block_path

    assignments_by_name = {item.name: item for item in assignments}
    expanded: dict[str, np.ndarray] = {}
    for name, raw_loop in waypoint_loops.items():
        assignment = assignments_by_name[name]
        loop = np.asarray(raw_loop, dtype=np.float64)
        points: list[np.ndarray] = [loop[0]]
        for first, second in zip(loop, np.roll(loop, -1, axis=0)):
            path = plan_block_path(
                regions,
                assignment.component_id,
                first,
                second,
                spacing_m=maximum_navigation_leg_m,
            )
            points.extend(path[1:])
        if len(points) > 1 and np.allclose(points[-1], points[0]):
            points.pop()
        result = np.asarray(points, dtype=np.float64)
        navigation_legs = np.linalg.norm(
            np.roll(result, -1, axis=0) - result, axis=1
        )
        if float(navigation_legs.max()) > maximum_navigation_leg_m + 1.0e-6:
            raise RuntimeError(
                f"expanded route for {name} exceeds navigation leg limit"
            )
        expanded[name] = result
    return expanded


class OfficialPeopleRoamingManager:
    """Re-issue seeded random goals through the official runtime on arrival."""

    def __init__(
        self,
        runtime: OfficialPeopleRuntime,
        regions: WalkableRegions,
        assignments: Sequence[RoamingAssignment],
        *,
        seed: int,
        arrival_radius_m: float = 0.65,
        minimum_trip_m: float = 5.0,
        maximum_trip_m: float = 30.0,
        waypoint_loops: Mapping[str, np.ndarray] | None = None,
        reassign_on_arrival: bool = True,
        dynamic_target_clearance_m: float = 3.0,
        dynamic_proximity_retarget_m: float = 2.5,
        people_target_clearance_m: float = 1.5,
        people_proximity_retarget_m: float = 2.2,
        people_yield_duration_s: float = 2.5,
        stall_retarget_after_s: float = 10.0,
    ) -> None:
        if len(assignments) != len(runtime.specs):
            raise ValueError("assignment/runtime character count mismatch")
        self.runtime = runtime
        self.regions = regions
        self.assignments = tuple(assignments)
        self.rng = np.random.default_rng(seed)
        self.arrival_radius_m = float(arrival_radius_m)
        self.minimum_trip_m = float(minimum_trip_m)
        self.maximum_trip_m = float(maximum_trip_m)
        self.reassign_on_arrival = bool(reassign_on_arrival)
        self.dynamic_target_clearance_m = float(dynamic_target_clearance_m)
        self.dynamic_proximity_retarget_m = float(dynamic_proximity_retarget_m)
        self.people_target_clearance_m = float(people_target_clearance_m)
        self.people_proximity_retarget_m = float(people_proximity_retarget_m)
        self.people_yield_duration_s = float(people_yield_duration_s)
        self.stall_retarget_after_s = float(stall_retarget_after_s)
        if self.people_yield_duration_s <= 0.0:
            raise ValueError("people_yield_duration_s must be positive")
        if self.stall_retarget_after_s <= 0.0:
            raise ValueError("stall_retarget_after_s must be positive")
        self.waypoint_loops = {
            name: np.asarray(points, dtype=np.float64)
            for name, points in (waypoint_loops or {}).items()
        }
        # -1 means the resident is still following its initial independently
        # sampled target.  On that first arrival the manager injects ring[0]
        # (the spawn point).  Later arrivals choose one of the two adjacent
        # pre-approved ring targets with the seeded RNG.  This is genuine
        # resident random target selection while keeping each leg short and
        # avoiding the known long Recast-query hang in Isaac Sim 4.5.
        self.ring_indices = {}
        for item in assignments:
            ring = self.waypoint_loops.get(item.name)
            initial_xy = np.asarray(item.initial_target_xyz[:2], dtype=np.float64)
            self.ring_indices[item.name] = (
                1
                if ring is not None
                and len(ring) > 1
                and np.allclose(initial_xy, ring[1])
                else -1
            )
        self.targets = np.asarray(
            [item.initial_target_xyz for item in assignments],
            dtype=np.float64,
        )
        self.completed_trips = np.zeros(len(assignments), dtype=np.int64)
        self.rejected_navmesh_targets = np.zeros(len(assignments), dtype=np.int64)
        # Guards against counting/injecting the same arrival every frame while a
        # character lingers inside ``arrival_radius_m``.  With
        # ``reassign_on_arrival=False`` it stays ``True`` after the one trip so
        # arrivals are counted exactly once without any in-loop Recast query.
        self.arrival_counted = np.zeros(len(assignments), dtype=bool)
        self.dynamic_proximity_redirects = np.zeros(len(assignments), dtype=np.int64)
        self.dynamic_emergency_redirects = np.zeros(len(assignments), dtype=np.int64)
        self.dynamic_proximity_active = np.zeros(len(assignments), dtype=bool)
        self.dynamic_proximity_obstacle_index = np.full(
            len(assignments), -1, dtype=np.int64
        )
        self.dynamic_proximity_level = np.zeros(len(assignments), dtype=np.int8)
        self.people_proximity_redirects = np.zeros(len(assignments), dtype=np.int64)
        self.people_proximity_yields = np.zeros(len(assignments), dtype=np.int64)
        self.people_proximity_active = np.zeros(len(assignments), dtype=bool)
        self.stall_anchor_positions_xy: np.ndarray | None = None
        self.stall_elapsed_s = np.zeros(len(assignments), dtype=np.float64)
        self.stall_retargets = np.zeros(len(assignments), dtype=np.int64)

    def _sample_queryable_target(self, index: int, current_xyz: np.ndarray) -> np.ndarray:
        component = self.assignments[index].component_id
        for _ in range(256):
            xy = self.regions.sample(component, self.rng)
            target = np.asarray((xy[0], xy[1], self.regions.config.ground_z_m), dtype=np.float64)
            trip_distance = float(np.linalg.norm(target[:2] - current_xyz[:2]))
            if trip_distance < self.minimum_trip_m or trip_distance > self.maximum_trip_m:
                continue
            route = self.runtime.navmesh.query_shortest_path(tuple(current_xyz), tuple(target))
            if route is not None and len(route.get_points()) >= 2:
                return target
            self.rejected_navmesh_targets[index] += 1
        raise RuntimeError(f"no queryable roaming target for {self.assignments[index].name}")

    def _ring_target(
        self,
        index: int,
        positions_xyz: np.ndarray,
        dynamic_obstacle_positions_xy: np.ndarray,
        *,
        avoid_from_xy: np.ndarray | None = None,
        avoid_person_xy: np.ndarray | None = None,
        allow_any_initial: bool = False,
    ) -> tuple[int, np.ndarray] | None:
        """Choose a seeded ring point that is not currently occupied.

        NVIDIA's NavigationManager deliberately ignores an obstacle when the
        final target is closer than that obstacle.  Avoiding occupied targets
        prevents a resident from being commanded through a stopped bicycle at
        the very end of a leg while leaving official path generation and
        avoidance untouched.
        """

        assignment = self.assignments[index]
        ring = self.waypoint_loops[assignment.name]
        current_index = self.ring_indices[assignment.name]
        if current_index < 0:
            # Normal first-arrival behavior returns to ring[0].  A proximity
            # escape may happen before that first arrival, in which case the
            # spawn point can be both too close and on the wrong side of the
            # other person; allow any approved ring point for that emergency.
            candidates = (
                list(range(len(ring)))
                if avoid_person_xy is not None or allow_any_initial
                else [0]
            )
        else:
            # Do not reverse or cut across a narrow NearRoad strip.  The ring
            # itself is seeded from random approved points; following its next
            # point preserves stochastic tasks without introducing a diagonal
            # chord through other residents.
            candidates = [
                (current_index + step) % len(ring)
                for step in range(1, len(ring))
            ]
            remaining = [value for value in range(len(ring)) if value not in candidates]
            self.rng.shuffle(remaining)
            candidates.extend(remaining)
        other_people_xy = np.delete(positions_xyz[:, :2], index, axis=0)
        other_targets_xy = np.delete(self.targets[:, :2], index, axis=0)
        for candidate_index in candidates:
            point = ring[candidate_index]
            # Occupancy filtering can skip the nominal next ring point and
            # expose another point that happens to sit inside the current
            # arrival disc.  Injecting such a target clears ``arrival_counted``
            # without making the character leave the disc, causing one GoTo
            # insertion per physics tick and eventually overflowing the native
            # CharacterBehavior queue.  A fresh task must require visible
            # translation beyond the arrival hysteresis.
            if float(np.linalg.norm(point - positions_xyz[index, :2])) < max(
                1.5, 2.0 * self.arrival_radius_m
            ):
                continue
            if (
                len(other_people_xy)
                and float(np.linalg.norm(other_people_xy - point, axis=1).min())
                < self.people_target_clearance_m
            ):
                continue
            # A currently empty endpoint can still be reserved by another
            # resident.  Without target-to-target admission two official
            # paths may converge onto adjacent ring points and deadlock only
            # after a long run, even though both targets were empty when
            # selected.
            if (
                len(other_targets_xy)
                and float(np.linalg.norm(other_targets_xy - point, axis=1).min())
                < self.people_target_clearance_m
            ):
                continue
            if (
                len(dynamic_obstacle_positions_xy)
                and float(
                    np.linalg.norm(dynamic_obstacle_positions_xy - point, axis=1).min()
                )
                < self.dynamic_target_clearance_m
            ):
                continue
            if avoid_from_xy is not None and len(dynamic_obstacle_positions_xy):
                obstacle_delta = dynamic_obstacle_positions_xy - avoid_from_xy
                nearest = int(np.argmin(np.linalg.norm(obstacle_delta, axis=1)))
                target_delta = point - avoid_from_xy
                # When a bicycle stops in the middle of the current route,
                # select a ring point in the opposite half-plane.  NVIDIA's
                # NavigationManager remains responsible for the actual path
                # and motion; this only avoids commanding a person to keep
                # walking toward a known near obstacle.
                if float(np.dot(target_delta, obstacle_delta[nearest])) > 0.0:
                    continue
            if avoid_person_xy is not None:
                # A narrow sidewalk may not leave NVIDIA's local avoidance
                # enough lateral room once two characters are already
                # approaching head-on.  Keep the official GoTo/NavMesh motion,
                # but choose a target in the half-plane away from the other
                # resident before that state becomes unrecoverable.
                target_delta = point - positions_xyz[index, :2]
                person_delta = avoid_person_xy - positions_xyz[index, :2]
                if float(np.dot(target_delta, person_delta)) > 0.0:
                    continue
            return candidate_index, point
        return None

    def _redirect_close_people(
        self,
        positions_xyz: np.ndarray,
        dynamic_obstacle_positions_xy: np.ndarray,
    ) -> set[int]:
        """Proactively separate both residents in each close same-block pair.

        This changes only the task target.  NVIDIA CharacterManager,
        NavigationManager and NavMesh remain the sole pose/path executors.
        Pair hysteresis prevents repeated command injection while the two
        characters separate.
        """

        redirected: set[int] = set()
        release_distance = self.people_proximity_retarget_m + 0.5
        # A character-level latch is intentional: pair-level hysteresis still
        # allows one resident in a three-person cluster to receive alternating
        # commands for two different pairs every frame.
        for index, assignment in enumerate(self.assignments):
            same_component = [
                other
                for other, candidate in enumerate(self.assignments)
                if other != index and candidate.component_id == assignment.component_id
            ]
            if not same_component:
                self.people_proximity_active[index] = False
                continue
            nearest = float(
                np.linalg.norm(
                    positions_xyz[same_component, :2] - positions_xyz[index, :2],
                    axis=1,
                ).min()
            )
            if nearest > release_distance:
                self.people_proximity_active[index] = False
        for first in range(len(positions_xyz)):
            for second in range(first + 1, len(positions_xyz)):
                if (
                    self.assignments[first].component_id
                    != self.assignments[second].component_id
                ):
                    continue
                distance = float(
                    np.linalg.norm(
                        positions_xyz[first, :2] - positions_xyz[second, :2]
                    )
                )
                if distance >= self.people_proximity_retarget_m:
                    continue
                # A one-sided yield is insufficient in a narrow strip: the
                # non-yielding resident can keep its old target beyond the
                # stopped person and walk directly into it.  Give both people
                # official GoTo targets in opposite half-planes.  One departs
                # immediately; the deterministic yielding side executes an
                # official Idle first, creating temporal right-of-way.
                yielding = second if (first + second) % 2 else first
                departing = first if yielding == second else second
                if (
                    yielding in redirected
                    or departing in redirected
                    or self.people_proximity_active[yielding]
                    or self.people_proximity_active[departing]
                ):
                    continue
                yielding_assignment = self.assignments[yielding]
                departing_assignment = self.assignments[departing]
                if (
                    yielding_assignment.name not in self.waypoint_loops
                    or departing_assignment.name not in self.waypoint_loops
                ):
                    continue
                departing_selected = self._ring_target(
                    departing,
                    positions_xyz,
                    dynamic_obstacle_positions_xy,
                    avoid_person_xy=positions_xyz[yielding, :2],
                )
                if departing_selected is None:
                    continue
                departing_index, departing_point = departing_selected
                # Reserve the immediate departure target while choosing the
                # yielding target; ``self.targets`` is committed only after
                # both choices succeed.
                yielding_dynamic_xy = np.asarray(
                    [*dynamic_obstacle_positions_xy, departing_point],
                    dtype=np.float64,
                )
                yielding_selected = self._ring_target(
                    yielding,
                    positions_xyz,
                    yielding_dynamic_xy,
                    avoid_person_xy=positions_xyz[departing, :2],
                )
                if yielding_selected is None:
                    continue
                yielding_index, yielding_point = yielding_selected
                self.ring_indices[departing_assignment.name] = departing_index
                self.ring_indices[yielding_assignment.name] = yielding_index
                departing_target = np.asarray(
                    (
                        departing_point[0],
                        departing_point[1],
                        self.regions.config.ground_z_m,
                    ),
                    dtype=np.float64,
                )
                yielding_target = np.asarray(
                    (
                        yielding_point[0],
                        yielding_point[1],
                        self.regions.config.ground_z_m,
                    ),
                    dtype=np.float64,
                )
                self.runtime.inject_goto(departing_assignment.name, departing_target)
                self.runtime.inject_yield_then_goto(
                    yielding_assignment.name,
                    yielding_target,
                    yield_duration_s=self.people_yield_duration_s,
                )
                self.targets[departing] = departing_target
                self.targets[yielding] = yielding_target
                self.arrival_counted[departing] = False
                self.arrival_counted[yielding] = False
                self.people_proximity_redirects[departing] += 1
                self.people_proximity_redirects[yielding] += 1
                self.people_proximity_yields[yielding] += 1
                self.people_proximity_active[departing] = True
                self.people_proximity_active[yielding] = True
                redirected.add(departing)
                redirected.add(yielding)
        return redirected

    def update(
        self,
        dynamic_obstacle_positions_xy: np.ndarray | Sequence[Sequence[float]] = (),
        *,
        delta_time_s: float = 0.02,
    ) -> np.ndarray:
        if float(delta_time_s) <= 0.0:
            raise ValueError("delta_time_s must be positive")
        positions = self.runtime.positions()
        dynamic_xy = np.asarray(dynamic_obstacle_positions_xy, dtype=np.float64)
        if dynamic_xy.size == 0:
            dynamic_xy = np.empty((0, 2), dtype=np.float64)
        elif dynamic_xy.ndim != 2 or dynamic_xy.shape[1] != 2:
            raise ValueError("dynamic obstacle positions must be an N x 2 array")
        people_redirected = self._redirect_close_people(positions, dynamic_xy)
        if self.stall_anchor_positions_xy is None:
            self.stall_anchor_positions_xy = positions[:, :2].copy()
        for index, (assignment, position) in enumerate(zip(self.assignments, positions)):
            if (
                float(
                    np.linalg.norm(
                        position[:2] - self.stall_anchor_positions_xy[index]
                    )
                )
                >= 0.25
            ):
                self.stall_anchor_positions_xy[index] = position[:2]
                self.stall_elapsed_s[index] = 0.0
                continue
            self.stall_elapsed_s[index] += float(delta_time_s)
            if (
                index in people_redirected
                or self.stall_elapsed_s[index] < self.stall_retarget_after_s
                or assignment.name not in self.waypoint_loops
            ):
                continue
            selected = self._ring_target(
                index,
                positions,
                dynamic_xy,
                allow_any_initial=True,
            )
            if selected is None:
                continue
            next_index, next_point = selected
            self.ring_indices[assignment.name] = next_index
            target = np.asarray(
                (next_point[0], next_point[1], self.regions.config.ground_z_m),
                dtype=np.float64,
            )
            self.runtime.inject_goto(assignment.name, target)
            self.targets[index] = target
            self.arrival_counted[index] = False
            self.stall_anchor_positions_xy[index] = position[:2]
            self.stall_elapsed_s[index] = 0.0
            self.stall_retargets[index] += 1
            people_redirected.add(index)
        for index, (assignment, position) in enumerate(zip(self.assignments, positions)):
            if index in people_redirected:
                continue
            target_blocked = False
            proximity_blocked = False
            proximity_emergency = False
            nearest_dynamic_index = -1
            if len(dynamic_xy):
                target_blocked = bool(
                    float(np.linalg.norm(dynamic_xy - self.targets[index, :2], axis=1).min())
                    < self.dynamic_target_clearance_m
                )
                dynamic_distances = np.linalg.norm(dynamic_xy - position[:2], axis=1)
                nearest_dynamic_index = int(np.argmin(dynamic_distances))
                proximity_distance = float(dynamic_distances[nearest_dynamic_index])
                if (
                    self.dynamic_proximity_active[index]
                    and self.dynamic_proximity_obstacle_index[index]
                    != nearest_dynamic_index
                ):
                    # A single boolean latch is insufficient in dense traffic:
                    # while a resident stays near one bicycle, a different
                    # bicycle can become the nearest threat.  Re-arm for the
                    # new official DynamicObstacle instead of silently
                    # carrying the old obstacle's hysteresis state forward.
                    self.dynamic_proximity_active[index] = False
                    self.dynamic_proximity_level[index] = 0
                if proximity_distance > self.dynamic_proximity_retarget_m + 0.5:
                    self.dynamic_proximity_active[index] = False
                    self.dynamic_proximity_obstacle_index[index] = -1
                    self.dynamic_proximity_level[index] = 0
                emergency_distance = max(
                    1.5, 0.65 * self.dynamic_proximity_retarget_m
                )
                proximity_emergency = bool(
                    proximity_distance < emergency_distance
                    and self.dynamic_proximity_level[index] < 2
                )
                proximity_blocked = bool(
                    (
                        proximity_distance < self.dynamic_proximity_retarget_m
                        and not self.dynamic_proximity_active[index]
                    )
                    or proximity_emergency
                )
            if (target_blocked or proximity_blocked) and assignment.name in self.waypoint_loops:
                selected = self._ring_target(
                    index,
                    positions,
                    dynamic_xy,
                    avoid_from_xy=position[:2] if proximity_blocked else None,
                )
                if selected is not None:
                    next_index, next_point = selected
                    self.ring_indices[assignment.name] = next_index
                    target = np.asarray(
                        (
                            next_point[0],
                            next_point[1],
                            self.regions.config.ground_z_m,
                        ),
                        dtype=np.float64,
                    )
                    self.runtime.inject_goto(assignment.name, target)
                    self.targets[index] = target
                    self.arrival_counted[index] = False
                    self.dynamic_proximity_redirects[index] += int(proximity_blocked)
                    self.dynamic_emergency_redirects[index] += int(
                        proximity_emergency
                    )
                    if proximity_blocked:
                        self.dynamic_proximity_active[index] = True
                        self.dynamic_proximity_obstacle_index[index] = (
                            nearest_dynamic_index
                        )
                        self.dynamic_proximity_level[index] = (
                            2 if proximity_emergency else 1
                        )
                    continue
            if np.linalg.norm(position[:2] - self.targets[index, :2]) > self.arrival_radius_m:
                continue
            if self.arrival_counted[index]:
                continue
            self.arrival_counted[index] = True
            self.completed_trips[index] += 1
            if not self.reassign_on_arrival:
                # Frozen single validated route: count the arrival but never
                # re-query Recast in the runtime loop (GIL-holding hang risk).
                continue
            if assignment.name in self.waypoint_loops:
                # Component-local bounded hop: no Python-side Recast query and
                # therefore no GIL-holding long-search failure.
                selected = self._ring_target(index, positions, dynamic_xy)
                if selected is None:
                    # Keep the completed command idle and retry next frame.
                    # Do not inject a target occupied by a person or bicycle.
                    self.completed_trips[index] -= 1
                    self.arrival_counted[index] = False
                    continue
                next_index, next_point = selected
                self.ring_indices[assignment.name] = next_index
                target = np.asarray(
                    (next_point[0], next_point[1], self.regions.config.ground_z_m),
                    dtype=np.float64,
                )
            else:
                target = self._sample_queryable_target(index, position)
            self.runtime.inject_goto(assignment.name, target)
            self.targets[index] = target
            self.arrival_counted[index] = False
        return positions

    def metrics(self) -> dict[str, object]:
        return {
            "resident_count": len(self.assignments),
            "completed_trips": self.completed_trips.tolist(),
            "rejected_navmesh_targets": self.rejected_navmesh_targets.tolist(),
            "dynamic_proximity_redirects": self.dynamic_proximity_redirects.tolist(),
            "dynamic_emergency_redirects": self.dynamic_emergency_redirects.tolist(),
            "people_proximity_redirects": self.people_proximity_redirects.tolist(),
            "people_proximity_yields": self.people_proximity_yields.tolist(),
            "people_yield_duration_s": self.people_yield_duration_s,
            "stall_retarget_after_s": self.stall_retarget_after_s,
            "stall_retargets": self.stall_retargets.tolist(),
            "component_ids": [item.component_id for item in self.assignments],
            "position_source": "official Animation Graph CharacterManager",
            "route_source": (
                "seeded random adjacent target from pre-approved component-local ring "
                "+ official runtime GoTo injection"
            ),
        }
