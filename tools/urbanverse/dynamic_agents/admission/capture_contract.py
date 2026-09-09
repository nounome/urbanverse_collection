"""Read-only structural audit of the existing Go2 three-camera frame index.

Passing this check proves file/record integrity, not pixel/physics synchronization,
camera calibration accuracy, successful locomotion, or joint-scene acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = ("camera_tm_pinhole", "camera_tm1_fisheye", "camera_tm2_fisheye")


def audit_capture_isolated(run_dir: Path) -> dict:
    """Do not decode PNG inside Kit's mixed extension/Pillow module context."""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    completed = subprocess.run([sys.executable, '-m',
        'urbanverse.dynamic_agents.admission.capture_contract', str(run_dir)],
        env=environment, capture_output=True, text=True, timeout=120)
    if completed.returncode not in (0, 1):
        raise RuntimeError(f'Isolated capture audit failed: {completed.stderr[-2000:]}')
    result = json.loads(completed.stdout)
    if completed.returncode != int(result['status'] == 'failed'):
        raise RuntimeError('Capture audit exit status disagrees with report')
    return result


def audit_capture(run_dir: Path) -> dict:
    root = Path(run_dir).resolve()
    errors, warnings = [], []
    count = 0
    previous_time = previous_step = -1
    sizes = {}
    invalid_depth_pixels = {name: 0 for name in CAMERAS}

    def asset_path(value):
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("asset path must be nonempty and relative to run")
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("asset path escapes run")
        return path

    index = root / "metadata/frame_index.jsonl"
    if not index.is_file():
        errors.append("missing metadata/frame_index.jsonl")
    else:
        with index.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                count += 1
                prefix = f"line {line_number}"
                try:
                    row = json.loads(line)
                    if row["frame_index"] != line_number - 1:
                        raise ValueError("noncontiguous zero-based frame_index")
                    timestamp = float(row["timestamp_s"])
                    step = row["step_index"]
                    if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_time:
                        raise ValueError("invalid/nonincreasing timestamp")
                    if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
                        raise ValueError("invalid/nonincreasing step_index")
                    previous_time, previous_step = timestamp, step
                    position = np.asarray(row["base_position_world"], dtype=float)
                    quat = np.asarray(row["base_quaternion_wxyz_world"], dtype=float)
                    if position.shape != (3,) or not np.isfinite(position).all():
                        raise ValueError("invalid Go2 position")
                    if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1) > .01:
                        raise ValueError("invalid Go2 quaternion")
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{prefix}: {exc}")
                    continue
                for name in CAMERAS:
                    try:
                        modalities = row["modalities"][name]
                        # Require the writer's exact frame basename: reusing one image
                        # for every index must not silently pass the structural check.
                        expected = f"frame_{row['frame_index']:04d}"
                        rgb_path = asset_path(modalities["rgb"])
                        depth_path = asset_path(modalities["depth_float32"])
                        if rgb_path.stem != expected or depth_path.stem != expected:
                            raise ValueError("asset frame number disagrees with index")
                        with Image.open(rgb_path) as image:
                            image.load()
                            if image.mode != "RGB":
                                raise ValueError("RGB image is not three-channel RGB")
                            shape = (image.height, image.width)
                        depth = np.load(depth_path, allow_pickle=False, mmap_mode="r")
                        if depth.dtype != np.float32 or depth.shape != shape:
                            raise ValueError("depth must be float32 and match RGB shape")
                        if name in sizes and sizes[name] != shape:
                            raise ValueError("resolution changed within capture")
                        sizes[name] = shape
                        if np.isnan(depth).any() or np.isneginf(depth).any() or (depth <= 0).any():
                            raise ValueError("depth contains NaN, negative infinity or nonpositive values")
                        invalid_depth_pixels[name] += int(np.isposinf(depth).sum())
                        if not np.isfinite(depth).any():
                            warnings.append(f"{prefix}/{name}: all depth invalid; inspect view coverage")
                    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                        errors.append(f"{prefix}/{name}: {exc}")
    if not count:
        errors.append("no captured frames")
    return {
        "schema_version": 1,
        "scope": "three_camera_file_and_frame_record_integrity_only",
        "status": "failed" if errors else "passed_with_warnings" if warnings else "passed",
        "frame_count": count,
        "camera_ids": list(CAMERAS),
        "invalid_depth_pixels": invalid_depth_pixels,
        "errors": errors,
        "warnings": warnings,
        "not_verified": ["pixel_timestamp_alignment", "calibration_and_extrinsics",
                         "depth_metric_accuracy", "motion_and_scene_acceptance",
                         "per_camera_overlap_labels"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit_capture(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
