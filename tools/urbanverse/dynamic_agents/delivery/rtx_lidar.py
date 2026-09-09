#!/usr/bin/env python3
"""Convert Isaac Sim 4.5 RTX LiDAR raw ticks into timestamped scans."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .coda_dataset import LidarScan


RAW_FIELDS = ("distances", "azimuths", "elevations", "intensities", "deltaTimes")


def _array(frame: Mapping[str, Any], name: str, dtype: Any) -> np.ndarray:
    value = np.asarray(frame.get(name, []), dtype=dtype).reshape(-1)
    return value


def raw_tick_to_arrays(frame: Mapping[str, Any], epoch_ns: int) -> dict[str, np.ndarray | int]:
    """Return positive finite hits in the native LiDAR frame.

    Isaac's raw node defines azimuth/elevation in degrees, ``timestampNs`` as
    the tick head, and ``deltaTimes`` as per-return nanoseconds from that head.
    ``channels`` is the physical channel and is delivered as PCD ``ring``.
    """
    arrays = {name: _array(frame, name, np.float64 if name != "deltaTimes" else np.uint64) for name in RAW_FIELDS}
    count = arrays["distances"].shape[0]
    if count == 0 or any(value.shape[0] != count for value in arrays.values()):
        raise ValueError("RTX LiDAR raw fields are empty or have inconsistent lengths")
    channels = _array(frame, "channels", np.uint32)
    if channels.shape[0] != count:
        channels = _array(frame, "emitterIds", np.uint32)
    if channels.shape[0] != count:
        raise ValueError("RTX LiDAR frame has no point-aligned channels/emitterIds")
    distances = arrays["distances"]
    azimuth = np.deg2rad(arrays["azimuths"])
    elevation = np.deg2rad(arrays["elevations"])
    valid = (
        np.isfinite(distances)
        & np.isfinite(azimuth)
        & np.isfinite(elevation)
        & np.isfinite(arrays["intensities"])
        & (distances > 0.0)
    )
    if not np.any(valid):
        raise ValueError("RTX LiDAR tick contains no positive finite returns")
    radial_xy = distances[valid] * np.cos(elevation[valid])
    xyz = np.column_stack(
        (
            radial_xy * np.cos(azimuth[valid]),
            radial_xy * np.sin(azimuth[valid]),
            distances[valid] * np.sin(elevation[valid]),
        )
    ).astype(np.float32)
    tick_timestamp_ns = int(frame.get("timestampNs", 0))
    point_timestamp_ns = (
        int(epoch_ns) + tick_timestamp_ns + arrays["deltaTimes"][valid].astype(np.uint64)
    ).astype(np.uint64)
    return {
        "xyz": xyz,
        "intensity": arrays["intensities"][valid].astype(np.float32),
        "ring": channels[valid].astype(np.uint32),
        "point_timestamp_ns": point_timestamp_ns,
        "frame_id": int(frame.get("frameId", tick_timestamp_ns)),
    }


class RtxLidarAccumulator:
    """Accumulate unique RTX render ticks into one delivery scan."""

    def __init__(self, epoch_ns: int) -> None:
        self.epoch_ns = int(epoch_ns)
        self._last_frame_id: int | None = None
        self._chunks: list[dict[str, np.ndarray | int]] = []

    @property
    def point_count(self) -> int:
        return sum(int(np.asarray(chunk["xyz"]).shape[0]) for chunk in self._chunks)

    def push(self, frame: Mapping[str, Any]) -> bool:
        chunk = raw_tick_to_arrays(frame, self.epoch_ns)
        frame_id = int(chunk["frame_id"])
        if frame_id == self._last_frame_id:
            return False
        self._last_frame_id = frame_id
        self._chunks.append(chunk)
        return True

    def pop_scan(self) -> LidarScan:
        if not self._chunks:
            raise RuntimeError("no RTX LiDAR ticks accumulated")
        xyz = np.concatenate([np.asarray(chunk["xyz"]) for chunk in self._chunks], axis=0)
        intensity = np.concatenate([np.asarray(chunk["intensity"]) for chunk in self._chunks])
        ring = np.concatenate([np.asarray(chunk["ring"]) for chunk in self._chunks])
        point_timestamp_ns = np.concatenate(
            [np.asarray(chunk["point_timestamp_ns"], dtype=np.uint64) for chunk in self._chunks]
        )
        self._chunks.clear()
        start = int(np.min(point_timestamp_ns))
        end = int(np.max(point_timestamp_ns))
        return LidarScan(
            xyz=xyz,
            intensity=intensity,
            ring=ring,
            point_timestamp_ns=point_timestamp_ns,
            timestamp_start_ns=start,
            timestamp_end_ns=end,
            reference_timestamp_ns=(start + end) // 2,
        ).validated()

    def pop_latest_scan(self) -> LidarScan:
        """Return only the newest render tick as one synchronized scan.

        RTX raw ticks can cover overlapping firing intervals.  Camera-rate
        delivery therefore uses the tick produced by the same render update
        as the RGB exposure instead of concatenating all intervening ticks.
        Older ticks are discarded when this method is called.
        """
        if not self._chunks:
            raise RuntimeError("no RTX LiDAR ticks accumulated")
        latest = self._chunks[-1]
        self._chunks.clear()
        xyz = np.asarray(latest["xyz"])
        intensity = np.asarray(latest["intensity"])
        ring = np.asarray(latest["ring"])
        point_timestamp_ns = np.asarray(latest["point_timestamp_ns"], dtype=np.uint64)
        start = int(np.min(point_timestamp_ns))
        end = int(np.max(point_timestamp_ns))
        return LidarScan(
            xyz=xyz,
            intensity=intensity,
            ring=ring,
            point_timestamp_ns=point_timestamp_ns,
            timestamp_start_ns=start,
            timestamp_end_ns=end,
            reference_timestamp_ns=(start + end) // 2,
        ).validated()
