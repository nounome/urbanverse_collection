#!/usr/bin/env python3
"""Reusable 2D pose math for recentering off-pivot UrbanVerse vehicles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rotation_2d(yaw_rad: float) -> np.ndarray:
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def transfer_front_heading_from_corresponding_points(
    source_points_xy: np.ndarray,
    proxy_points_xy: np.ndarray,
    source_front_heading_rad: float,
    *,
    end_fraction: float = 0.08,
) -> float:
    """Transfer a signed vehicle-front direction through a visual proxy.

    ``source_points_xy`` and ``proxy_points_xy`` must contain corresponding
    mesh vertices in the same order.  Selecting the front/rear ends in the
    already calibrated source frame and observing those exact vertices in the
    proxy frame preserves both the long-axis rotation and the front/rear sign,
    even when a GLB payload has model-specific root transforms.
    """
    source = np.asarray(source_points_xy, dtype=np.float64)
    proxy = np.asarray(proxy_points_xy, dtype=np.float64)
    if source.shape != proxy.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(
            f"expected matching Nx2 point arrays, got {source.shape} and {proxy.shape}"
        )
    if len(source) < 4:
        raise ValueError("at least four corresponding mesh vertices are required")
    if not 0.0 < end_fraction <= 0.25:
        raise ValueError(f"end_fraction must be in (0, 0.25], got {end_fraction}")
    forward = np.asarray(
        [math.cos(source_front_heading_rad), math.sin(source_front_heading_rad)],
        dtype=np.float64,
    )
    order = np.argsort(source @ forward)
    end_count = max(2, int(math.ceil(len(source) * end_fraction)))
    rear_indices = order[:end_count]
    front_indices = order[-end_count:]
    proxy_axis = proxy[front_indices].mean(axis=0) - proxy[rear_indices].mean(axis=0)
    norm = float(np.linalg.norm(proxy_axis))
    if norm <= 1.0e-8:
        raise ValueError("proxy front/rear landmark separation is degenerate")
    return math.atan2(float(proxy_axis[1]), float(proxy_axis[0]))


def transform_footprint(
    local_corners_xy: list[list[float]],
    center_xy: list[float],
    yaw_rad: float,
) -> np.ndarray:
    """Transform body-centered footprint corners to a requested world pose."""
    corners = np.asarray(local_corners_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    return corners @ rotation_2d(yaw_rad).T + center


def recentered_root_matrix_2d(
    original_root_matrix: np.ndarray,
    original_body_center_xy: list[float],
    target_body_center_xy: list[float],
    delta_yaw_rad: float,
) -> np.ndarray:
    """Return a root matrix that rotates geometry about its visible body center.

    Matrices use column-vector convention.  The formula is
    ``T(target_center) R(delta_yaw) T(-original_center) M_original``.
    It is suitable for an authored root whose origin is far from its mesh.
    """
    original = np.asarray(original_root_matrix, dtype=np.float64)
    if original.shape != (4, 4):
        raise ValueError(f"expected 4x4 root matrix, got {original.shape}")
    source_center = np.asarray(original_body_center_xy, dtype=np.float64)
    target_center = np.asarray(target_body_center_xy, dtype=np.float64)
    before = np.eye(4, dtype=np.float64)
    before[:2, 3] = -source_center
    rotate = np.eye(4, dtype=np.float64)
    rotate[:2, :2] = rotation_2d(delta_yaw_rad)
    after = np.eye(4, dtype=np.float64)
    after[:2, 3] = target_center
    return after @ rotate @ before @ original


def calibration_pose(
    record: dict[str, Any], center_xy: list[float], signed_heading_deg: float
) -> dict[str, Any]:
    """Create footprint and ground-contact pose data from one registry record."""
    if record["front_direction"]["status"] == "pending_visual_validation":
        raise ValueError(f"front direction is not validated: {record['asset_id']}")
    initial_heading = float(record["front_direction"]["heading_deg"])
    target_heading = math.radians(signed_heading_deg)
    corners = transform_footprint(record["footprint"]["body_corners_xy"], center_xy, target_heading)
    return {
        "center_xy": [float(value) for value in center_xy],
        "signed_heading_deg": float(signed_heading_deg),
        "delta_yaw_deg": float(signed_heading_deg - initial_heading),
        "footprint_corners_xy": corners.tolist(),
        "body_center_height_above_support_m": float(record["grounding"]["body_center_height_above_support_m"]),
    }
