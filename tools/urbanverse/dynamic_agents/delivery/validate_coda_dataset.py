#!/usr/bin/env python3
"""Validate a CoDa multi-sensor delivery directory without launching Isaac."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .coda_dataset import CAMERA_NAMES


PCD_FIELDS = "FIELDS x y z intensity ring point_timestamp_ns"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_dataset(
    root: Path, sequence_name: str = "seq_000", require_three_pinhole: bool = False
) -> dict[str, Any]:
    root = Path(root)
    sequence = root / "sequences" / sequence_name
    errors: list[str] = []
    camera_rows: dict[str, list[dict[str, str]]] = {}
    resolutions: set[tuple[int, int]] = set()
    maximum_black_border_ratio = 0.0
    pinhole_intrinsics: dict[str, dict[str, Any]] = {}
    for name in CAMERA_NAMES:
        for required in (
            root / "calibration" / "intrinsics" / f"{name}.yaml",
            root / "calibration" / "extrinsics" / f"T_body_{name}.txt",
        ):
            if not required.is_file():
                errors.append(f"missing {required}")
        intrinsics_path = root / "calibration" / "intrinsics" / f"{name}.yaml"
        if require_three_pinhole and intrinsics_path.is_file():
            pinhole_intrinsics[name] = yaml.safe_load(intrinsics_path.read_text(encoding="utf-8"))
        index = sequence / "images" / name / "frames.csv"
        if not index.is_file():
            errors.append(f"missing {index}")
            camera_rows[name] = []
            continue
        rows = _rows(index)
        camera_rows[name] = rows
        timestamps = [int(row["timestamp_ns"]) for row in rows]
        if len(timestamps) != len(set(timestamps)) or timestamps != sorted(timestamps):
            errors.append(f"{name} timestamps are duplicated or not increasing")
        for row in rows:
            image_path = sequence / "images" / name / row["filename"]
            if not image_path.is_file() or image_path.stem != row["timestamp_ns"]:
                errors.append(f"invalid image index entry: {image_path}")
                continue
            # Isaac Sim 4.5's process can leave Pillow's PngImageFile without
            # the private _close_fp attribute expected by its context manager.
            # Loading eagerly also lets Pillow release the PNG file handle.
            image = Image.open(image_path)
            image.load()
            resolutions.add(image.size)
            if require_three_pinhole:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                edges = (rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1])
                edge_black_ratios = [
                    float(np.mean(np.max(edge, axis=1) <= 1)) for edge in edges
                ]
                border = np.concatenate(edges, axis=0)
                black_ratio = float(np.mean(np.max(border, axis=1) <= 1))
                maximum_black_border_ratio = max(maximum_black_border_ratio, black_ratio)
                # A dark vehicle may legitimately touch one or two image edges.
                # A projection mask instead affects all four edges (fisheye), or
                # makes at least one complete edge black (letter/pillar boxing).
                likely_projection_mask = min(edge_black_ratios) > 0.05 or max(
                    edge_black_ratios
                ) > 0.95
                if likely_projection_mask:
                    errors.append(
                        f"{name} image has likely invalid black border/mask "
                        f"({black_ratio:.3f}, edges={edge_black_ratios}): {image_path}"
                    )
    counts = {name: len(rows) for name, rows in camera_rows.items()}
    if len(set(counts.values())) != 1 or not counts or min(counts.values(), default=0) == 0:
        errors.append(f"camera frame counts differ or are empty: {counts}")
    if len(resolutions) != 1:
        errors.append(f"camera resolutions differ: {sorted(resolutions)}")
    if require_three_pinhole and len(resolutions) == 1:
        width, height = next(iter(resolutions))
        if width < 960 or height < 640:
            errors.append(f"pinhole resolution is below 960x640: {width}x{height}")
        reference_k = None
        for name in CAMERA_NAMES:
            calibration = pinhole_intrinsics.get(name, {})
            if calibration.get("camera_model") != "pinhole":
                errors.append(f"{name} camera_model is not pinhole")
            if calibration.get("distortion_model") != "opencv_radtan":
                errors.append(f"{name} distortion_model is not opencv_radtan")
            distortion = calibration.get("distortion", {})
            if not distortion or any(abs(float(value)) > 1.0e-12 for value in distortion.values()):
                errors.append(f"{name} ideal-pinhole distortion is not all zero: {distortion}")
            intrinsics = calibration.get("intrinsics", {})
            try:
                k = tuple(float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
            except (KeyError, TypeError, ValueError):
                errors.append(f"{name} has invalid pinhole intrinsics: {intrinsics}")
                continue
            if reference_k is None:
                reference_k = k
            elif not np.allclose(k, reference_k, rtol=0.0, atol=1.0e-9):
                errors.append(f"{name} intrinsics differ from the shared pinhole rig")
            horizontal_fov = math.degrees(2.0 * math.atan(float(width) / (2.0 * k[0])))
            if not 80.0 - 1.0e-6 <= horizontal_fov <= 100.0 + 1.0e-6:
                errors.append(f"{name} horizontal FOV is outside 80-100 degrees: {horizontal_fov}")

        yaws: dict[str, float] = {}
        for name in CAMERA_NAMES:
            path = root / "calibration" / "extrinsics" / f"T_body_{name}.txt"
            if not path.is_file():
                continue
            matrix = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
            optical_forward_body = matrix[:3, 2]
            yaws[name] = math.degrees(
                math.atan2(float(optical_forward_body[1]), float(optical_forward_body[0]))
            )
        if len(yaws) == 3:
            if abs(yaws["cam_front"]) > 1.0:
                errors.append(f"cam_front is not forward-facing: yaw={yaws['cam_front']}")
            if not (40.0 <= yaws["cam_front_left"] <= 60.0):
                errors.append(f"cam_front_left yaw is not forward-left: {yaws['cam_front_left']}")
            if not (-60.0 <= yaws["cam_front_right"] <= -40.0):
                errors.append(f"cam_front_right yaw is not forward-right: {yaws['cam_front_right']}")
            horizontal_fov = math.degrees(
                2.0 * math.atan(float(width) / (2.0 * float(reference_k[0])))
            ) if reference_k is not None else 0.0
            for first, second in (("cam_front_left", "cam_front"), ("cam_front", "cam_front_right")):
                separation = abs(yaws[first] - yaws[second])
                overlap = (horizontal_fov - separation) / horizontal_fov
                if not 0.20 - 1.0e-6 <= overlap <= 0.40 + 1.0e-6:
                    errors.append(f"{first}/{second} angular overlap is outside 20%-40%: {overlap}")

        timestamp_lists = {
            name: [int(row["timestamp_ns"]) for row in camera_rows.get(name, [])]
            for name in CAMERA_NAMES
        }
        if any(values != timestamp_lists[CAMERA_NAMES[0]] for values in timestamp_lists.values()):
            errors.append("three pinhole cameras do not have exactly identical timestamps")

    scans_path = sequence / "lidar" / "scans.csv"
    scans = _rows(scans_path) if scans_path.is_file() else []
    if not scans:
        errors.append("LiDAR scan index is missing or empty")
    lidar_reference_times: list[int] = []
    for row in scans:
        start = int(row["timestamp_start_ns"])
        end = int(row["timestamp_end_ns"])
        reference = int(row["reference_timestamp_ns"])
        lidar_reference_times.append(reference)
        if not start <= reference <= end:
            errors.append(f"LiDAR reference outside interval: {row}")
        pcd = sequence / "lidar" / row["filename"]
        if not pcd.is_file() or pcd.stem != row["reference_timestamp_ns"]:
            errors.append(f"invalid PCD index entry: {pcd}")
            continue
        header = pcd.read_bytes().split(b"DATA binary\n", 1)[0].decode("ascii", errors="replace")
        if PCD_FIELDS not in header or "TYPE F F F F U U" not in header:
            errors.append(f"PCD fields/types do not meet contract: {pcd}")

    poses_path = sequence / "ego_pose.csv"
    poses = _rows(poses_path) if poses_path.is_file() else []
    pose_times = {int(row["timestamp_ns"]) for row in poses}
    front_times = [int(row["timestamp_ns"]) for row in camera_rows.get("cam_front", [])]
    missing_pose_times = sorted(set(front_times) - pose_times)
    if missing_pose_times:
        errors.append(f"{len(missing_pose_times)} camera timestamps have no exact ego pose")
    maximum_sync_offset_ns = None
    if front_times and lidar_reference_times:
        maximum_sync_offset_ns = max(
            min(abs(camera_time - lidar_time) for lidar_time in lidar_reference_times)
            for camera_time in front_times
        )
        if maximum_sync_offset_ns > 10_000_000:
            errors.append(f"camera-LiDAR reference offset exceeds 10 ms: {maximum_sync_offset_ns} ns")

    for required in (
        root / "calibration" / "extrinsics" / "T_body_lidar.txt",
        root / "calibration" / "time_offsets.yaml",
        sequence / "sequence_info.yaml",
    ):
        if not required.is_file():
            errors.append(f"missing {required}")
    return {
        "status": "success" if not errors else "failed",
        "dataset_root": str(root),
        "sequence_name": sequence_name,
        "camera_frame_counts": counts,
        "lidar_scan_count": len(scans),
        "ego_pose_count": len(poses),
        "resolution": list(next(iter(resolutions))) if len(resolutions) == 1 else None,
        "maximum_camera_lidar_reference_offset_ns": maximum_sync_offset_ns,
        "three_pinhole_required": bool(require_three_pinhole),
        "maximum_black_border_ratio": maximum_black_border_ratio if require_three_pinhole else None,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--sequence", default="seq_000")
    parser.add_argument("--require-three-pinhole", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_dataset(args.dataset_root, args.sequence, args.require_three_pinhole)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
