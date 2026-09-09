"""Per-step evidence aggregation independent of Isaac/torch imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StepMetricsAccumulator:
    maximum_step_displacement_m: float
    _positions: list[np.ndarray] = field(default_factory=list)
    _actions: list[np.ndarray] = field(default_factory=list)
    _foot_support_steps: int = 0
    _nonfoot_steps: int = 0
    _nonfinite_steps: int = 0
    _discontinuity_steps: list[int] = field(default_factory=list)
    _max_step_displacement: float = 0.0
    _max_action_delta: float = 0.0

    def update(
        self,
        step_index: int,
        position_xyz: list[float] | np.ndarray,
        action: list[float] | np.ndarray,
        contact_classification: dict[str, Any],
    ) -> dict[str, Any]:
        position = np.asarray(position_xyz, dtype=np.float64)
        action_array = np.asarray(action, dtype=np.float64)
        finite = bool(np.isfinite(position).all() and np.isfinite(action_array).all())
        self._nonfinite_steps += int(not finite)
        displacement = 0.0
        if self._positions:
            displacement = float(np.linalg.norm(position - self._positions[-1]))
            self._max_step_displacement = max(self._max_step_displacement, displacement)
            if displacement > self.maximum_step_displacement_m:
                self._discontinuity_steps.append(int(step_index))
        action_delta = 0.0
        if self._actions:
            action_delta = float(np.linalg.norm(action_array - self._actions[-1]))
            self._max_action_delta = max(self._max_action_delta, action_delta)
        self._positions.append(position)
        self._actions.append(action_array)
        supporting_feet = contact_classification.get("supporting_feet", [])
        self._foot_support_steps += int(bool(supporting_feet))
        self._nonfoot_steps += int(bool(contact_classification.get("non_foot_collision", False)))
        return {
            "step_displacement_m": displacement,
            "policy_action_delta_l2": action_delta,
            "finite_state_and_action": finite,
            "trajectory_continuous": displacement <= self.maximum_step_displacement_m,
        }

    def summary(self) -> dict[str, Any]:
        count = len(self._positions)
        path_length = float(sum(
            np.linalg.norm(self._positions[index] - self._positions[index - 1])
            for index in range(1, count)
        ))
        action_matrix = np.asarray(self._actions, dtype=np.float64) if self._actions else np.empty((0, 0))
        action_span = (
            float(np.max(np.ptp(action_matrix, axis=0)))
            if action_matrix.size
            else 0.0
        )
        return {
            "step_count": count,
            "path_length_m": path_length,
            "foot_support_step_count": self._foot_support_steps,
            "foot_support_fraction": self._foot_support_steps / max(1, count),
            "nonfoot_contact_step_count": self._nonfoot_steps,
            "nonfoot_contact_fraction": self._nonfoot_steps / max(1, count),
            "nonfinite_step_count": self._nonfinite_steps,
            "maximum_step_displacement_m": self._max_step_displacement,
            "trajectory_discontinuity_step_indices": self._discontinuity_steps,
            "trajectory_continuous": not self._discontinuity_steps and self._nonfinite_steps == 0,
            "policy_action_sample_count": len(self._actions),
            "policy_action_dimension": int(action_matrix.shape[1]) if action_matrix.ndim == 2 else 0,
            "maximum_policy_action_delta_l2": self._max_action_delta,
            "maximum_policy_action_component_span": action_span,
            "policy_action_varied": bool(action_span > 1.0e-5),
        }
