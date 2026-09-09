"""Reciprocal speed limits for project-controlled mixed dynamic agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MixedAgentState:
    """One scene-independent circle used by the shared local safety layer."""

    agent_id: str
    kind: str
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_xy, dtype=np.float64)
        velocity = np.asarray(self.velocity_xy, dtype=np.float64)
        if position.shape != (2,) or velocity.shape != (2,):
            raise ValueError("mixed-agent position and velocity must be xy vectors")
        if float(self.radius_m) <= 0.0:
            raise ValueError("mixed-agent radius must be positive")
        object.__setattr__(self, "position_xy", position)
        object.__setattr__(self, "velocity_xy", velocity)


def reciprocal_speed_scales(
    agents: Sequence[MixedAgentState],
    *,
    prediction_horizon_s: float = 1.5,
    minimum_clearance_m: float = 0.15,
    slow_clearance_m: float = 1.5,
    audit: MutableMapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return symmetric longitudinal speed scales for every pair.

    The result never invents lateral motion.  Pedestrians may apply the scale
    to their path tangent and non-holonomic agents apply it to forward speed.
    A separate exact footprint guard remains responsible for the final commit.
    """

    if prediction_horizon_s <= 0.0:
        raise ValueError("prediction_horizon_s must be positive")
    if minimum_clearance_m < 0.0 or slow_clearance_m <= minimum_clearance_m:
        raise ValueError("mixed-agent clearance thresholds are invalid")
    ids = [agent.agent_id for agent in agents]
    if len(ids) != len(set(ids)):
        raise ValueError("mixed-agent ids must be unique")
    scales = {agent.agent_id: 1.0 for agent in agents}
    for first_index, first in enumerate(agents):
        for second in agents[first_index + 1 :]:
            pair_kind = "--".join(sorted((first.kind, second.kind)))
            pair_audit = None
            if audit is not None:
                pair_audit = audit.setdefault(
                    pair_kind,
                    {
                        "evaluated_pair_steps": 0,
                        "limited_pair_steps": 0,
                        "full_stop_pair_steps": 0,
                        "minimum_predicted_clearance_m": None,
                    },
                )
                pair_audit["evaluated_pair_steps"] += 1
            relative = second.position_xy - first.position_xy
            relative_velocity = second.velocity_xy - first.velocity_xy
            combined_radius = float(first.radius_m + second.radius_m)
            current_clearance = float(np.linalg.norm(relative)) - combined_radius
            speed_sq = float(np.dot(relative_velocity, relative_velocity))
            closest_time = (
                float(
                    np.clip(
                        -np.dot(relative, relative_velocity) / speed_sq,
                        0.0,
                        prediction_horizon_s,
                    )
                )
                if speed_sq > 1.0e-12
                else 0.0
            )
            predicted_clearance = (
                float(np.linalg.norm(relative + relative_velocity * closest_time))
                - combined_radius
            )
            clearance = min(current_clearance, predicted_clearance)
            if pair_audit is not None:
                previous_minimum = pair_audit["minimum_predicted_clearance_m"]
                pair_audit["minimum_predicted_clearance_m"] = (
                    float(clearance)
                    if previous_minimum is None
                    else min(float(previous_minimum), float(clearance))
                )
            if clearance >= slow_clearance_m:
                continue
            scale = float(
                np.clip(
                    (clearance - minimum_clearance_m)
                    / (slow_clearance_m - minimum_clearance_m),
                    0.0,
                    1.0,
                )
            )
            # Both participants receive the same limit.  Their concrete
            # controllers remain different, but neither class is passive.
            scales[first.agent_id] = min(scales[first.agent_id], scale)
            scales[second.agent_id] = min(scales[second.agent_id], scale)
            if pair_audit is not None:
                pair_audit["limited_pair_steps"] += 1
                pair_audit["full_stop_pair_steps"] += int(scale <= 1.0e-9)
    return scales
