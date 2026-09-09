#!/usr/bin/env python3
"""Deterministic calibration helpers for document-aligned camera captures.

The route capture renders an undistorted/native base image, then this module
maps RGB and ray-distance depth into the requested OpenCV pixel domain.  This
keeps the exact document K/D values in the output rather than merely recording
them as metadata.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def rotation_matrix_rz_ry_rx(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def quaternion_wxyz_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return a normalized Hamilton quaternion for a 3x3 rotation matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
            )
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat *= -1.0
    return tuple(float(value) for value in quat)


def camera_ext(calibration: dict[str, Any]) -> dict[str, Any]:
    rotation = rotation_matrix_rz_ry_rx(calibration["rpy_vehicle_camera_rad"])
    translation = np.asarray(calibration["position_vehicle_xyz_m"], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {
        "T_vehicle_camera": transform.tolist(),
        "quaternion_wxyz_vehicle_camera": list(quaternion_wxyz_from_matrix(rotation)),
        "position_vehicle_xyz_m": translation.tolist(),
        "rpy_vehicle_camera_rad": [float(value) for value in calibration["rpy_vehicle_camera_rad"]],
        "direction": "camera_to_vehicle",
        "frame": "ROS optical",
        "euler_order": "Rz(yaw)*Ry(pitch)*Rx(roll)",
    }


def _opencv_fisheye_inverse_map(calibration: dict[str, Any], width: int, height: int):
    k = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(calibration["D"], dtype=np.float64)
    yy, xx = np.indices((height, width), dtype=np.float64)
    xd = (xx - k[0, 2]) / k[0, 0]
    yd = (yy - k[1, 2]) / k[1, 1]
    theta_d = np.sqrt(xd * xd + yd * yd)
    theta_limit = math.radians(95.0)
    limit2 = theta_limit * theta_limit
    projected_limit = theta_limit * (
        1.0 + d[0] * limit2 + d[1] * limit2**2 + d[2] * limit2**3 + d[3] * limit2**4
    )
    angular_valid = theta_d <= projected_limit
    lower = np.zeros_like(theta_d)
    upper = np.full_like(theta_d, theta_limit)
    for _ in range(48):
        theta = 0.5 * (lower + upper)
        t2 = theta * theta
        projected_mid = theta * (1.0 + d[0] * t2 + d[1] * t2**2 + d[2] * t2**3 + d[3] * t2**4)
        lower = np.where(projected_mid < theta_d, theta, lower)
        upper = np.where(projected_mid >= theta_d, theta, upper)
    theta = 0.5 * (lower + upper)
    direction_x = np.divide(xd, theta_d, out=np.zeros_like(xd), where=theta_d > 1e-12)
    direction_y = np.divide(yd, theta_d, out=np.zeros_like(yd), where=theta_d > 1e-12)
    source_focal_px = float(k[0, 0])
    map_x = k[0, 2] + source_focal_px * theta * direction_x
    map_y = k[1, 2] + source_focal_px * theta * direction_y
    t2 = theta * theta
    projected = theta * (1.0 + d[0] * t2 + d[1] * t2**2 + d[2] * t2**3 + d[3] * t2**4)
    residual = np.abs(projected - theta_d)
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (theta >= 0.0)
        & angular_valid
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid, {
        "model": "OpenCV fisheye theta polynomial",
        "source_model": "undistorted equidistant fisheyePolynomial",
        "max_inverse_residual_normalized": float(np.max(residual[valid])) if np.any(valid) else None,
        "valid_output_ratio": float(np.mean(valid)),
    }


def _opencv_radtan_inverse_map(calibration: dict[str, Any], width: int, height: int):
    """Map a four-coefficient pinhole image using k1,k2,p1,p2.

    The source document labels the centre block as fisheye while its camera ID
    and project requirement call it pinhole.  The strict run records this
    explicit pinhole interpretation rather than hiding the ambiguity.
    """
    k = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
    k1, k2, p1, p2 = (float(value) for value in calibration["D"])
    yy, xx = np.indices((height, width), dtype=np.float64)
    xd = (xx - k[0, 2]) / k[0, 0]
    yd = (yy - k[1, 2]) / k[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(32):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - delta_x) / radial
        y = (yd - delta_y) / radial
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    check_x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    check_y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    residual = np.sqrt((check_x - xd) ** 2 + (check_y - yd) ** 2)
    map_x = k[0, 2] + k[0, 0] * x
    map_y = k[1, 2] + k[1, 1] * y
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid, {
        "model": "OpenCV pinhole radtan k1,k2,p1,p2",
        "source_model": "undistorted pinhole",
        "document_model_ambiguity": "centre block says fisheye; camera ID and project text say pinhole",
        "max_inverse_residual_normalized": float(np.max(residual[valid])) if np.any(valid) else None,
        "valid_output_ratio": float(np.mean(valid)),
    }


def build_inverse_map(calibration: dict[str, Any], width: int, height: int):
    if [int(width), int(height)] != [int(value) for value in calibration["image_size"]]:
        raise ValueError(
            f"strict calibration requires native image size {calibration['image_size']}, got {[width, height]}"
        )
    if calibration["role"] == "center_forward_pinhole":
        return _opencv_radtan_inverse_map(calibration, width, height)
    return _opencv_fisheye_inverse_map(calibration, width, height)


def remap_rgb_depth(
    rgb: np.ndarray, depth: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear RGB plus nearest-neighbour ray-depth remapping."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    height, width = map_x.shape
    if rgb.shape[:2] != (height, width) or depth.shape != (height, width):
        raise ValueError(f"shape mismatch rgb={rgb.shape} depth={depth.shape} map={(height, width)}")
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    wx = map_x - x0
    wy = map_y - y0
    output_rgb = np.zeros_like(rgb)
    blended = (
        rgb[y0, x0].astype(np.float32) * ((1.0 - wx) * (1.0 - wy))[..., None]
        + rgb[y0, x1].astype(np.float32) * (wx * (1.0 - wy))[..., None]
        + rgb[y1, x0].astype(np.float32) * ((1.0 - wx) * wy)[..., None]
        + rgb[y1, x1].astype(np.float32) * (wx * wy)[..., None]
    )
    output_rgb[valid] = np.clip(np.rint(blended[valid]), 0.0, 255.0).astype(np.uint8)
    nearest_x = np.clip(np.rint(map_x).astype(np.int32), 0, width - 1)
    nearest_y = np.clip(np.rint(map_y).astype(np.int32), 0, height - 1)
    output_depth = np.full((height, width), np.inf, dtype=np.float32)
    output_depth[valid] = depth[nearest_y[valid], nearest_x[valid]]
    return output_rgb, output_depth
