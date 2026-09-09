"""Restricted non-holonomic bicycle motion and calibrated pose math."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .calibration import AssetCalibration


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class MotionLimits:
    maximum_speed_mps: float
    maximum_acceleration_mps2: float = 1.0
    maximum_braking_mps2: float = 2.0
    maximum_steering_rate_radps: float = 0.8

    def validate(self) -> None:
        if min(
            self.maximum_speed_mps,
            self.maximum_acceleration_mps2,
            self.maximum_braking_mps2,
            self.maximum_steering_rate_radps,
        ) <= 0.0:
            raise ValueError("motion limits must be positive")


@dataclass(frozen=True)
class MicromobilityState:
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float = 0.0
    steering_rad: float = 0.0
    distance_m: float = 0.0

    @property
    def position_xy(self) -> np.ndarray:
        return np.asarray([self.x_m, self.y_m], dtype=np.float64)

    @property
    def velocity_xy(self) -> np.ndarray:
        return np.asarray(
            [math.cos(self.yaw_rad), math.sin(self.yaw_rad)], dtype=np.float64
        ) * self.speed_mps


def step_bicycle(
    state: MicromobilityState,
    *,
    desired_speed_mps: float,
    desired_steering_rad: float,
    dt_s: float,
    calibration: AssetCalibration,
    limits: MotionLimits,
) -> MicromobilityState:
    """Advance a forward-only bicycle model; reverse and lateral commands do not exist."""
    limits.validate()
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    target_speed = float(np.clip(desired_speed_mps, 0.0, limits.maximum_speed_mps))
    delta = target_speed - state.speed_mps
    rate = limits.maximum_acceleration_mps2 if delta >= 0.0 else limits.maximum_braking_mps2
    speed = max(0.0, state.speed_mps + float(np.clip(delta, -rate * dt_s, rate * dt_s)))
    target_steering = float(
        np.clip(
            desired_steering_rad,
            -calibration.maximum_steering_rad,
            calibration.maximum_steering_rad,
        )
    )
    steering = state.steering_rad + float(
        np.clip(
            target_steering - state.steering_rad,
            -limits.maximum_steering_rate_radps * dt_s,
            limits.maximum_steering_rate_radps * dt_s,
        )
    )
    curvature = math.tan(steering) / calibration.wheelbase_m
    curvature = float(
        np.clip(
            curvature,
            -1.0 / calibration.minimum_turning_radius_m,
            1.0 / calibration.minimum_turning_radius_m,
        )
    )
    midpoint_yaw = state.yaw_rad + 0.5 * speed * curvature * dt_s
    displacement = speed * dt_s * np.asarray(
        [math.cos(midpoint_yaw), math.sin(midpoint_yaw)], dtype=np.float64
    )
    return replace(
        state,
        x_m=state.x_m + float(displacement[0]),
        y_m=state.y_m + float(displacement[1]),
        yaw_rad=wrap_angle(state.yaw_rad + speed * curvature * dt_s),
        speed_mps=speed,
        steering_rad=steering,
        distance_m=state.distance_m + float(np.linalg.norm(displacement)),
    )


def footprint_corners(
    state: MicromobilityState, calibration: AssetCalibration, margin_m: float = 0.0
) -> np.ndarray:
    half_length = calibration.length_m * 0.5 + margin_m
    half_width = calibration.width_m * 0.5 + margin_m
    local = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + state.position_xy


def visible_center_and_root_pose(
    state: MicromobilityState,
    calibration: AssetCalibration,
    ground_z_m: float,
) -> dict[str, object]:
    """Map a calibrated visible-center state to its converted USD root pose."""
    local_front_angle = math.atan2(*reversed(calibration.converted_front_axis_xy))
    root_yaw = wrap_angle(state.yaw_rad - local_front_angle)
    center_z = ground_z_m + calibration.height_m * 0.5
    root_z = ground_z_m + calibration.ground_offset_m
    return {
        "visible_center_xyz_m": [state.x_m, state.y_m, center_z],
        "root_translation_xyz_m": [state.x_m, state.y_m, root_z],
        "root_yaw_rad": root_yaw,
    }


def heading_velocity_error(state: MicromobilityState) -> float:
    if state.speed_mps <= 1.0e-9:
        return 0.0
    velocity_yaw = math.atan2(float(state.velocity_xy[1]), float(state.velocity_xy[0]))
    return abs(wrap_angle(velocity_yaw - state.yaw_rad))


def ground_contact_error(
    root_z_m: float, ground_z_m: float, calibration: AssetCalibration
) -> float:
    return float(root_z_m - calibration.ground_offset_m - ground_z_m)
