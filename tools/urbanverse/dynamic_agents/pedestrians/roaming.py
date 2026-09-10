"""Fixed-population official People roaming within approved connected blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
from typing import Mapping, Sequence

import numpy as np

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
