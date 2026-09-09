#!/usr/bin/env python3
"""Write the CoDa-4DGS multi-sensor delivery contract.

The simulation clock is the authoritative common sensor clock.  A sequence
epoch anchors that clock in Unix nanoseconds so filenames remain ordinary
19-digit timestamps while all relative timing stays deterministic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image


CAMERA_NAMES = ("cam_front", "cam_front_left", "cam_front_right")


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list, tuple)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_yaml_lines(payload)) + "\n", encoding="utf-8")


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"expected finite 4x4 transform, got {matrix.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.12g")


@dataclass(frozen=True)
class LidarScan:
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    point_timestamp_ns: np.ndarray
    timestamp_start_ns: int
    timestamp_end_ns: int
    reference_timestamp_ns: int

    def validated(self) -> "LidarScan":
        xyz = np.asarray(self.xyz, dtype=np.float32)
        intensity = np.asarray(self.intensity, dtype=np.float32).reshape(-1)
        ring = np.asarray(self.ring, dtype=np.uint32).reshape(-1)
        point_time = np.asarray(self.point_timestamp_ns, dtype=np.uint64).reshape(-1)
        count = xyz.shape[0]
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be Nx3, got {xyz.shape}")
        if any(array.shape[0] != count for array in (intensity, ring, point_time)):
            raise ValueError("LiDAR fields have inconsistent point counts")
        if count == 0 or not np.isfinite(xyz).all() or not np.isfinite(intensity).all():
            raise ValueError("LiDAR scan must contain finite points")
        if not (int(self.timestamp_start_ns) <= int(self.reference_timestamp_ns) <= int(self.timestamp_end_ns)):
            raise ValueError("LiDAR reference timestamp is outside the scan interval")
        if np.any(point_time < int(self.timestamp_start_ns)) or np.any(point_time > int(self.timestamp_end_ns)):
            raise ValueError("per-point timestamp is outside the scan interval")
        return LidarScan(
            xyz=xyz,
            intensity=intensity,
            ring=ring,
            point_timestamp_ns=point_time,
            timestamp_start_ns=int(self.timestamp_start_ns),
            timestamp_end_ns=int(self.timestamp_end_ns),
            reference_timestamp_ns=int(self.reference_timestamp_ns),
        )


def write_binary_pcd(path: Path, scan: LidarScan) -> None:
    scan = scan.validated()
    points = np.empty(
        scan.xyz.shape[0],
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("intensity", "<f4"),
                ("ring", "<u4"),
                ("point_timestamp_ns", "<u8"),
            ]
        ),
    )
    points["x"], points["y"], points["z"] = scan.xyz.T
    points["intensity"] = scan.intensity
    points["ring"] = scan.ring
    points["point_timestamp_ns"] = scan.point_timestamp_ns
    count = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity ring point_timestamp_ns\n"
        "SIZE 4 4 4 4 4 8\n"
        "TYPE F F F F U U\n"
        "COUNT 1 1 1 1 1 1\n"
        f"WIDTH {count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {count}\n"
        "DATA binary\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(points.tobytes(order="C"))


class CodaDatasetWriter:
    """Incrementally write one synchronized sequence without overwriting data."""

    def __init__(
        self,
        root: Path,
        sequence_name: str,
        epoch_ns: int,
        camera_rate_hz: float,
        lidar_rate_hz: float,
        pose_source: str = "Isaac Sim ground-truth T_world_body",
    ) -> None:
        self.root = Path(root)
        self.sequence_name = sequence_name
        self.epoch_ns = int(epoch_ns)
        self.camera_rate_hz = float(camera_rate_hz)
        self.lidar_rate_hz = float(lidar_rate_hz)
        self.pose_source = pose_source
        self.sequence_dir = self.root / "sequences" / sequence_name
        if self.sequence_dir.exists() and any(self.sequence_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite sequence: {self.sequence_dir}")
        self._camera_rows: dict[str, list[list[Any]]] = {name: [] for name in CAMERA_NAMES}
        self._scan_rows: list[list[Any]] = []
        self._pose_rows: list[list[Any]] = []
        self._timestamps: list[int] = []
        for name in CAMERA_NAMES:
            (self.sequence_dir / "images" / name).mkdir(parents=True, exist_ok=True)
        (self.sequence_dir / "lidar").mkdir(parents=True, exist_ok=True)

    def timestamp_ns(self, simulation_time_s: float) -> int:
        return self.epoch_ns + int(round(float(simulation_time_s) * 1_000_000_000.0))

    def write_calibration(
        self,
        camera_calibrations: Mapping[str, Mapping[str, Any]],
        body_camera_transforms: Mapping[str, np.ndarray],
        body_lidar_transform: np.ndarray,
    ) -> None:
        if set(camera_calibrations) != set(CAMERA_NAMES):
            raise ValueError(f"calibration camera names must be {CAMERA_NAMES}")
        for name in CAMERA_NAMES:
            calibration = camera_calibrations[name]
            k = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
            d = [float(value) for value in calibration["D"]]
            model = str(calibration["camera_model"])
            distortion_model = str(calibration["distortion_model"])
            payload = {
                "camera_name": name,
                "image_width": int(calibration["image_width"]),
                "image_height": int(calibration["image_height"]),
                "camera_model": model,
                "distortion_model": distortion_model,
                "intrinsics": {
                    "fx": float(k[0, 0]),
                    "fy": float(k[1, 1]),
                    "cx": float(k[0, 2]),
                    "cy": float(k[1, 2]),
                },
                "distortion": {f"k{index + 1}": value for index, value in enumerate(d)},
            }
            if distortion_model == "opencv_radtan" and len(d) >= 4:
                payload["distortion"] = {"k1": d[0], "k2": d[1], "p1": d[2], "p2": d[3], "k3": d[4] if len(d) > 4 else 0.0}
            write_yaml(self.root / "calibration" / "intrinsics" / f"{name}.yaml", payload)
            write_matrix(
                self.root / "calibration" / "extrinsics" / f"T_body_{name}.txt",
                np.asarray(body_camera_transforms[name]),
            )
        write_matrix(self.root / "calibration" / "extrinsics" / "T_body_lidar.txt", body_lidar_transform)
        write_yaml(
            self.root / "calibration" / "time_offsets.yaml",
            {
                "reference_clock": "isaac_simulation_clock",
                "timestamp_unit": "ns",
                "timestamp_definition": "camera exposure midpoint; LiDAR reference is scan midpoint",
                "offset_definition": "t_reference = t_sensor + offset_ns",
                "offsets_ns": {**{name: 0 for name in CAMERA_NAMES}, "lidar": 0},
                "epoch_ns": self.epoch_ns,
            },
        )

    def add_camera_frame(
        self,
        camera_name: str,
        frame_index: int,
        timestamp_ns: int,
        rgb: np.ndarray,
        exposure_us: int = 0,
        gain: float = 1.0,
    ) -> Path:
        if camera_name not in self._camera_rows:
            raise ValueError(f"unknown camera: {camera_name}")
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB must be HxWx3, got {rgb.shape}")
        filename = f"{int(timestamp_ns)}.png"
        path = self.sequence_dir / "images" / camera_name / filename
        if path.exists():
            raise FileExistsError(path)
        Image.fromarray(rgb, mode="RGB").save(path)
        self._camera_rows[camera_name].append(
            [int(frame_index), int(timestamp_ns), filename, int(exposure_us), float(gain)]
        )
        self._timestamps.append(int(timestamp_ns))
        return path

    def add_lidar_scan(self, scan_index: int, scan: LidarScan) -> Path:
        scan = scan.validated()
        filename = f"{scan.reference_timestamp_ns}.pcd"
        path = self.sequence_dir / "lidar" / filename
        if path.exists():
            raise FileExistsError(path)
        write_binary_pcd(path, scan)
        self._scan_rows.append(
            [scan_index, scan.timestamp_start_ns, scan.timestamp_end_ns, scan.reference_timestamp_ns, filename]
        )
        self._timestamps.extend([scan.timestamp_start_ns, scan.timestamp_end_ns])
        return path

    def add_ego_pose(self, timestamp_ns: int, position_xyz: Iterable[float], quaternion_wxyz: Iterable[float]) -> None:
        position = np.asarray(list(position_xyz), dtype=np.float64)
        quat_wxyz = np.asarray(list(quaternion_wxyz), dtype=np.float64)
        if position.shape != (3,) or quat_wxyz.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quat_wxyz).all():
            raise ValueError("pose must contain finite xyz and wxyz quaternion")
        norm = float(np.linalg.norm(quat_wxyz))
        if norm <= 1.0e-12:
            raise ValueError("pose quaternion has zero norm")
        w, x, y, z = quat_wxyz / norm
        self._pose_rows.append([int(timestamp_ns), *position.tolist(), x, y, z, w])
        self._timestamps.append(int(timestamp_ns))

    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _observed_rate_hz(
        rows: list[list[Any]], timestamp_column: int, requested_rate_hz: float
    ) -> float:
        timestamps = np.asarray([int(row[timestamp_column]) for row in rows], dtype=np.int64)
        if timestamps.size < 2:
            return float(requested_rate_hz)
        intervals = np.diff(timestamps)
        if np.any(intervals <= 0):
            raise ValueError("sensor timestamps must be strictly increasing")
        return float(1_000_000_000.0 / np.median(intervals))

    def finalize(self, known_issues: list[str] | None = None) -> None:
        for name, rows in self._camera_rows.items():
            self._write_csv(
                self.sequence_dir / "images" / name / "frames.csv",
                ["frame_index", "timestamp_ns", "filename", "exposure_us", "gain"],
                rows,
            )
        self._write_csv(
            self.sequence_dir / "lidar" / "scans.csv",
            ["scan_index", "timestamp_start_ns", "timestamp_end_ns", "reference_timestamp_ns", "filename"],
            self._scan_rows,
        )
        self._write_csv(
            self.sequence_dir / "ego_pose.csv",
            ["timestamp_ns", "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"],
            self._pose_rows,
        )
        if not self._timestamps:
            raise RuntimeError("cannot finalize an empty sequence")
        camera_rate_hz = self._observed_rate_hz(
            self._camera_rows[CAMERA_NAMES[0]], 1, self.camera_rate_hz
        )
        lidar_rate_hz = self._observed_rate_hz(self._scan_rows, 3, self.lidar_rate_hz)
        write_yaml(
            self.sequence_dir / "sequence_info.yaml",
            {
                "sequence_name": self.sequence_name,
                "start_timestamp_ns": min(self._timestamps),
                "end_timestamp_ns": max(self._timestamps),
                "camera_rate_hz": camera_rate_hz,
                "lidar_rate_hz": lidar_rate_hz,
                "requested_camera_rate_hz": self.camera_rate_hz,
                "requested_lidar_rate_hz": self.lidar_rate_hz,
                "pose_source": self.pose_source,
                "known_issues": known_issues or [],
            },
        )
