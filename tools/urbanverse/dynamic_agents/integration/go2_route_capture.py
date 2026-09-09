#!/usr/bin/env python3
"""Capture complete, timestamp-aligned Go2 route runs in UrbanVerse scenes.

This is a data-generation qualification tool, not a locomotion-training tool.
It uses an existing published Isaac Lab Go2 controller as a black-box executor
and retains full raw runs rather than imposing a downstream training schema.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from appearance_randomization import isaac45_urbanverse_appearance_randomization_smoke as appearance  # noqa: E402
from urbanverse.dynamic_agents.navigation import control as navigation  # noqa: E402
from urbanverse.dynamic_agents.navigation import global_route_planner as global_planner  # noqa: E402
from urbanverse.dynamic_agents.navigation.joint_route_constraints import (  # noqa: E402
    build_joint_route_constraints,
)
from urbanverse.dynamic_agents.vehicles import phase1_dynamic_vehicle as phase1_dynamic  # noqa: E402
from simulation_qualification.isaac45_urbanverse_sensor_suite import (  # noqa: E402
    CAMERA_CALIBRATIONS,
    calibration_for_resolution,
    camera_intrinsics,
)
from urbanverse.dynamic_agents.rendering import camera_calibration as strict_calibration  # noqa: E402
from urbanverse.dynamic_agents.traffic.continuous_manager import (  # noqa: E402
    Scene10ContinuousVehicleManager,
)
from urbanverse.dynamic_agents.traffic.multivehicle_manager import (  # noqa: E402
    INTEGRATED_MIXED_VEHICLE_IDS,
)
from urbanverse.dynamic_agents.config import TrafficSceneConfig  # noqa: E402
from urbanverse.dynamic_agents.delivery import CodaDatasetWriter, RtxLidarAccumulator  # noqa: E402
from urbanverse.dynamic_agents.delivery.validate_coda_dataset import validate_dataset  # noqa: E402
from urbanverse.dynamic_agents.integration.scene_setup import (  # noqa: E402
    author_portable_traffic_visuals,
    configure_go2_support_corridor,
    configure_preauthored_traffic_visuals,
)
from urbanverse.dynamic_agents.pedestrians import (  # noqa: E402
    OfficialPeopleManager,
    author_official_people,
    configure_preauthored_people,
)
from urbanverse.dynamic_agents.navigation.policy_selection import DEFAULT_POLICY_KIND, EXTERNAL_SOURCE, EXTERNAL_KINDS
from urbanverse.viz_style import load_font  # noqa: E402


def safe_print(*args: Any, **kwargs: Any) -> bool:
    """Best-effort logging: a closed parent pipe must not change run status."""
    try:
        print(*args, **kwargs)
        return True
    except (BrokenPipeError, OSError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("craftbench", "training"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--turn-sign", type=float, choices=(-1.0, 1.0), required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--wrapper-template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("torchscript", *EXTERNAL_KINDS), default=DEFAULT_POLICY_KIND)
    parser.add_argument("--external-policy-source", type=Path, default=EXTERNAL_SOURCE)
    parser.add_argument(
        "--locomotion-profile",
        choices=tuple(navigation.CONTROLLER_PROFILES),
        default="flat",
        help="Isaac Lab environment profile; external policies require flat.",
    )
    parser.add_argument(
        "--rough-height-scan-ground-z-stage-units",
        type=float,
        help=(
            "Override the Rough policy height scanner with an invisible, non-colliding flat mesh at this "
            "world-Z value. This is only a policy observation proxy; scene contacts and rendering remain unchanged."
        ),
    )
    parser.add_argument(
        "--rough-height-scan-extent-stage-units",
        type=float,
        default=1000.0,
        help="Half extent of the optional Rough-policy height scan proxy.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("policy", "trajectory_carrier"),
        default="policy",
        help=(
            "policy uses the frozen controller for physical base motion; trajectory_carrier prescribes the "
            "articulated Go2 base pose on a prevalidated road while retaining synchronized sensors and raw data."
        ),
    )
    parser.add_argument(
        "--trajectory-carrier-height-offset-m",
        type=float,
        default=0.0,
        help="Raise the prescribed Go2/sensor carrier above its admitted ground pose; intended for aerial data capture.",
    )
    parser.add_argument(
        "--trajectory-carrier-speed",
        type=float,
        help="Override the prescribed trajectory-carrier speed in metres per second.",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--sensor-width", type=int, default=960)
    parser.add_argument("--sensor-height", type=int, default=768)
    parser.add_argument("--overview-width", type=int, default=1280)
    parser.add_argument("--overview-height", type=int, default=720)
    parser.add_argument("--overview-chase-distance", type=float, default=2.0)
    parser.add_argument("--overview-chase-lateral-offset", type=float, default=2.8)
    parser.add_argument("--overview-chase-height", type=float, default=9.0)
    parser.add_argument("--overview-target-forward", type=float, default=1.2)
    parser.add_argument("--overview-target-height", type=float, default=0.25)
    parser.add_argument("--overview-focal-length", type=float, default=12.0)
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument(
        "--coda-delivery",
        action="store_true",
        help="Also write a CoDa-4DGS dataset under <run-dir>/coda_dataset.",
    )
    parser.add_argument(
        "--coda-pinhole-rig",
        action="store_true",
        help="Use three synchronized undistorted pinhole cameras for CoDa delivery and preview.",
    )
    parser.add_argument("--coda-pinhole-horizontal-fov-deg", type=float, default=80.0)
    parser.add_argument("--coda-pinhole-side-yaw-deg", type=float, default=50.0)
    parser.add_argument("--coda-sequence-name", default="seq_000")
    parser.add_argument(
        "--coda-epoch-ns",
        type=int,
        help="Unix-nanosecond anchor for simulation time zero; defaults to the run start wall clock.",
    )
    parser.add_argument(
        "--rtx-lidar-config",
        default="OS1_REV6_32ch10hz1024res",
        help="Isaac Sim RTX LiDAR config basename; CoDa delivery requires the raw RTX LiDAR stream.",
    )
    parser.add_argument(
        "--rtx-lidar-mount-xyz",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.35),
        metavar=("X", "Y", "Z"),
        help="LiDAR origin in the Go2 body frame, metres.",
    )
    parser.add_argument(
        "--dynamic-agent-motion-blur",
        action="store_true",
        help="Enable RTX motion blur for dynamic-agent captures; disabled by default.",
    )
    parser.add_argument(
        "--reference-route-file",
        type=Path,
        help="Execute points_xy from a previously validated world-XY reference-route JSON.",
    )
    parser.add_argument(
        "--external-route-lookahead-distance",
        type=float,
        default=1.20,
        help="Pure-pursuit lookahead used with --reference-route-file.",
    )
    parser.add_argument(
        "--external-route-minimum-tracking-speed",
        type=float,
        default=0.24,
        help="Minimum forward speed used with --reference-route-file.",
    )
    parser.add_argument(
        "--lightweight-three-panel",
        action="store_true",
        help=(
            "Render only centre depth and third-person RGB, skip onboard RGB/raw depth, and build "
            "a trajectory/depth/third-person review video."
        ),
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Review-video speed multiplier; 2.0 encodes 2 fps captures at 4 fps.",
    )
    parser.add_argument(
        "--collection-light-scale",
        type=float,
        default=1.0,
        help="Scale every composed USD light intensity in the run-local wrapper; source USD is unchanged.",
    )
    parser.add_argument(
        "--collection-exposure-ev",
        type=float,
        default=0.0,
        help="Set the RTX tonemap exposure for all captured RGB images.",
    )
    parser.add_argument(
        "--source-dome-background-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep source DomeLight backgrounds visible in primary camera rays.",
    )
    parser.add_argument(
        "--source-dome-texture-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use source DomeLight HDR textures; disable to keep a uniform visible background.",
    )
    parser.add_argument("--collection-dome-light-scale", type=float)
    parser.add_argument("--collection-distant-light-scale", type=float)
    parser.add_argument("--collection-sphere-light-scale", type=float)
    parser.add_argument(
        "--collection-strong-sphere-light-scale",
        type=float,
        help="Override SphereLights at or above --collection-strong-sphere-intensity-threshold.",
    )
    parser.add_argument("--collection-strong-sphere-intensity-threshold", type=float, default=10000.0)
    parser.add_argument(
        "--collection-light-path-scale",
        action="append",
        default=[],
        metavar="PRIM_PATH=SCALE",
        help="Exact prim-path intensity override; repeat for multiple selected lights.",
    )
    parser.add_argument("--route-duration-scale", type=float, default=1.0)
    parser.add_argument(
        "--spawn-forward-offset",
        type=float,
        default=0.0,
        help=(
            "Move the derived spawn by this recorded distance along its admitted road heading. "
            "This is a runtime route-placement override and never edits the source USD."
        ),
    )
    parser.add_argument(
        "--spawn-world-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Explicit recorded road-route spawn XY; the source USD remains unchanged.",
    )
    parser.add_argument(
        "--spawn-world-yaw-deg",
        type=float,
        help="Absolute world yaw used with --spawn-world-xy.",
    )
    parser.add_argument(
        "--road-route-extent",
        type=float,
        help=(
            "Override only the admitted straight-road route extent for a short preview. "
            "The override is recorded and does not modify the shared scene catalog."
        ),
    )
    parser.add_argument(
        "--right-angle-route",
        action="store_true",
        help="Admit and follow a two-leg road route with an approximately 90-degree turn.",
    )
    parser.add_argument(
        "--right-angle-corner-distances",
        type=float,
        nargs="+",
        default=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5),
        help="Candidate first-leg lengths searched for a collision-free road corner.",
    )
    parser.add_argument(
        "--right-angle-second-leg-extent",
        type=float,
        default=4.0,
        help="Required road length after the admitted left/right 90-degree turn.",
    )
    parser.add_argument(
        "--right-angle-turn-deg",
        type=float,
        choices=(-90.0, 90.0),
        help=(
            "Force the second leg to turn -90 degrees (clockwise/right) or +90 degrees "
            "(counter-clockwise/left) from the admitted first-leg heading. By default both "
            "directions are probed and the longest admitted route is selected."
        ),
    )
    parser.add_argument(
        "--route-duration-seconds",
        type=float,
        help=(
            "Run the live-pose route controller for this simulation duration. "
            "This is intended for 2-5 minute raw runs."
        ),
    )
    parser.add_argument(
        "--global-route",
        action="store_true",
        help=(
            "Use the global reference-route planner (grid + ESDF + A* + spline + dense "
            "PhysX corridor re-validation) instead of the fixed-direction corridor probe."
        ),
    )
    parser.add_argument(
        "--global-route-seed",
        type=int,
        help="Seeded RNG for endpoint sampling in the global route planner (default from scene config).",
    )
    parser.add_argument(
        "--global-route-extent",
        type=float,
        help=(
            "Override the grid half-extent in metres for the global route planner; "
            "recorded and does not modify the shared scene catalog."
        ),
    )
    parser.add_argument(
        "--global-route-target-length",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        help=(
            "Override the target route-length band in metres (e.g. '1.5 2.5'). When absent "
            "and --route-duration-seconds is set, the band is calibrated from measured gait speed."
        ),
    )
    parser.add_argument(
        "--global-route-min-turn",
        type=float,
        metavar="RAD",
        help=(
            "Enforce a minimum cumulative heading change (rad) along the admitted global "
            "route, measured on the delivered spawn-prepended waypoint polyline."
        ),
    )
    parser.add_argument(
        "--global-route-cache-dir",
        type=Path,
        help="Directory for the grid+ESDF disk cache (default cache/global_route).",
    )
    parser.add_argument(
        "--safe-shuttle",
        action="store_true",
        help=(
            "Walk forward along prevalidated road spokes, turn in admitted endpoint clearance, "
            "and return forward through the same corridor."
        ),
    )
    parser.add_argument(
        "--omit-motion-vectors",
        action="store_true",
        help="Do not request/store motion vectors; RGB, float depth, poses, controls and contacts remain unchanged.",
    )
    parser.add_argument(
        "--phase1-dynamic-vehicle",
        action="store_true",
        help="Run one original scene vehicle across the route and enable synchronized phase-one evidence.",
    )
    parser.add_argument(
        "--phase1-avoidance-mode",
        choices=("yield", "disabled"),
        default="yield",
        help="Go2-owned stop/yield/resume controller, or a diagnostic no-avoidance challenge run.",
    )
    parser.add_argument(
        "--phase1-vehicle-prim",
        default="/World/vehicle_private_vehicle_suv_d90c7f830f9c41398bb55de4a2e001be",
        help="Original scene kinematic vehicle root to schedule across the Go2 route.",
    )
    parser.add_argument("--phase1-crossing-time-s", type=float, default=6.0)
    parser.add_argument("--phase1-vehicle-distance", type=float, default=12.0)
    parser.add_argument("--phase1-vehicle-speed", type=float, default=2.0)
    parser.add_argument(
        "--scene10-continuous-traffic",
        "--continuous-traffic",
        dest="scene10_continuous_traffic",
        action="store_true",
        help=(
            "Run calibrated vehicles with continuous safe spawn, "
            "forward bicycle motion, longitudinal CPU ORCA caps and junction reservation."
        ),
    )
    parser.add_argument(
        "--traffic-scene-config",
        type=Path,
        help="Portable scene JSON containing road, vehicle catalog and automotive routes.",
    )
    parser.add_argument("--traffic-vehicle-count", type=int, default=5)
    parser.add_argument("--traffic-initial-fill", action="store_true")
    parser.add_argument("--traffic-registry", type=Path)
    parser.add_argument("--traffic-audit-inventory", type=Path)
    parser.add_argument("--traffic-validated-routes", type=Path)
    parser.add_argument("--traffic-automotive-routes", type=Path)
    parser.add_argument("--minimum-go2-vehicle-clearance", type=float, default=1.0)
    parser.add_argument("--maximum-traffic-visual-center-error", type=float, default=0.1)
    parser.add_argument("--maximum-rendered-body-axis-error", type=float, default=3.0)
    parser.add_argument("--minimum-visible-vehicle-pass-duration", type=float, default=3.0)
    parser.add_argument(
        "--derived-visualization-stride",
        type=int,
        default=1,
        help="Save depth PNG visualizations every N captured frames; float32 depth is always saved.",
    )
    parser.add_argument(
        "--full-metrics-stride",
        type=int,
        default=1,
        help="Compute full-resolution RGB/depth metrics every N captured frames; other frames use a recorded spatial sample.",
    )
    parser.add_argument(
        "--strict-document-calibration",
        action="store_true",
        help=(
            "Use document CameraExt and apply the recorded OpenCV D to RGB/depth. "
            "By default this requires the native 1920x1536 resolution."
        ),
    )
    parser.add_argument(
        "--allow-scaled-document-calibration",
        action="store_true",
        help=(
            "With strict document calibration, allow the same-aspect-ratio render resolution and "
            "scale fx/fy/cx/cy into that pixel domain. D and CameraExt are unchanged and both the "
            "original and scaled K are recorded."
        ),
    )
    parser.add_argument("--warmup-updates", type=int, default=16)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument(
        "--appearance-profile",
        choices=(
            "baseline",
            "light_bright_noon",
            "light_warm_sunset",
            "ground_material_cool_glossy",
            "object_material_magenta",
            "camera_wide_bright",
        ),
        default="baseline",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=appearance.json_default) + "\n",
        encoding="utf-8",
    )


def measured_gait_speed_mps(scene_slug: str) -> float | None:
    """Best measured Go2 horizontal gait speed from the newest recorded run trajectory."""
    outputs_root = PROJECT_ROOT / "outputs" / "rtx3090_isaac45" / "go2_navigation_capture"
    if not outputs_root.exists():
        return None
    trajectories = sorted(
        outputs_root.glob(f"run_*/scene_runs/{scene_slug}/replicate_*/metadata/trajectory.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in trajectories:
        try:
            positions: list[list[float]] = []
            timestamps: list[float] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    base = row.get("base_position_world")
                    timestamp = row.get("timestamp_s")
                    if base is None or timestamp is None:
                        continue
                    positions.append([float(base[0]), float(base[1])])
                    timestamps.append(float(timestamp))
            if len(positions) < 3:
                continue
            points = np.asarray(positions, dtype=np.float64)
            times = np.asarray(timestamps, dtype=np.float64)
            span_s = float(times[-1] - times[0])
            travelled_m = float(np.linalg.norm(points[-1] - points[0]))
            # Ignore failed/short runs (stuck or immediate abort) so their near-zero
            # speeds do not poison the route-length calibration.
            if span_s < 3.0 or travelled_m < 0.5:
                continue
            step_dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
            step_dt = np.diff(times)
            valid = (step_dt > 1e-6) & np.isfinite(step_dist)
            speeds = step_dist[valid] / step_dt[valid]
            speeds = speeds[np.isfinite(speeds) & (speeds > 0.0)]
            if len(speeds) == 0:
                continue
            # Robust estimate: median of per-step speeds during active locomotion.
            return float(np.median(speeds))
        except Exception:
            continue
    return None


def calibrate_global_route_length(
    scene_slug: str,
    global_route_config: dict[str, Any],
    route_duration_seconds: float | None,
    max_forward_speed_mps: float,
) -> list[float] | None:
    """Derive the target route-length band (metres) from the measured gait speed.

    The commanded max_forward_speed overestimates the effective gait speed, so the
    band is calibrated from the measured trajectory speed when available and falls
    back to ``max_forward_speed * 0.6`` otherwise. Settle/stop overhead (~2 s) is
    subtracted from the requested duration so the robot walks for most of the run.
    """
    if route_duration_seconds is None:
        return None
    duration = float(route_duration_seconds)
    measured = measured_gait_speed_mps(scene_slug)
    if measured is None or not math.isfinite(measured) or measured <= 0.0:
        measured = float(max_forward_speed_mps) * 0.6
    walking_duration = max(1.0, duration - 2.0)
    # Scale the centre cap with the requested duration: 45 s keeps the historical 8 m
    # cap, longer runs (60 s) reach 10 m so the route fills the whole clip while short
    # smoke runs (30 s) drop to 6 m to stay on admitted turning routes.
    center_cap = max(4.0, 8.0 + 2.0 * (duration - 45.0) / 15.0)
    center = min(center_cap, max(1.8, measured * walking_duration))
    band = [round(center * 0.85, 2), round(center * 1.15, 2)]
    if band[0] <= 0.0 or band[1] < band[0]:
        band = [1.5, 2.5]
    return band


def navigation_frame_metrics(rgb: np.ndarray, depth: np.ndarray, full_resolution: bool) -> dict[str, Any]:
    """Validate every frame without making long high-resolution runs metric-bound."""
    if full_resolution:
        result = appearance.image_metrics(rgb, depth)
        result["sampling"] = {"mode": "full_resolution", "spatial_stride": 1}
        return result
    spatial_stride = max(2, int(math.ceil(max(rgb.shape[:2]) / 320.0)))
    sampled_rgb = rgb[::spatial_stride, ::spatial_stride]
    sampled_depth = depth[::spatial_stride, ::spatial_stride]
    result = appearance.image_metrics(sampled_rgb, sampled_depth)
    result["sampling"] = {
        "mode": "regular_spatial_sample",
        "spatial_stride": spatial_stride,
        "source_shape": list(rgb.shape),
        "sample_shape": list(sampled_rgb.shape),
    }
    return result


def save_depth_visualization(depth: np.ndarray, vis_path: Path) -> None:
    """Write a derived preview; callers separately preserve every float32 array."""
    finite = np.isfinite(depth)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        sampled = depth[::8, ::8]
        sampled = sampled[np.isfinite(sampled)]
        if sampled.size:
            lo, hi = np.percentile(sampled, [2.0, 98.0])
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            vis[finite] = ((1.0 - norm[finite]) * 255.0).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(vis_path)


def append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, default=appearance.json_default) + "\n")
    handle.flush()


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def author_all_light_scale(
    wrapper_path: Path,
    scale: float,
    *,
    dome_scale: float | None = None,
    distant_scale: float | None = None,
    sphere_scale: float | None = None,
    strong_sphere_scale: float | None = None,
    strong_sphere_intensity_threshold: float = 10000.0,
    path_scales: dict[str, float] | None = None,
    source_dome_background_visible: bool = True,
    source_dome_texture_enabled: bool = True,
) -> dict[str, Any]:
    """Author base and optional per-class intensity multipliers in a run-local wrapper."""
    from pxr import Sdf, Usd, UsdLux

    requested_scales = {
        "base": scale,
        "DomeLight": dome_scale,
        "DistantLight": distant_scale,
        "SphereLight": sphere_scale,
        "strong_SphereLight": strong_sphere_scale,
    }
    path_scales = dict(path_scales or {})
    if any(value is not None and value <= 0.0 for value in requested_scales.values()):
        raise ValueError("collection light scales must be positive")
    if strong_sphere_intensity_threshold < 0.0:
        raise ValueError("strong SphereLight intensity threshold must be non-negative")
    if any(not path.startswith("/") or value <= 0.0 for path, value in path_scales.items()):
        raise ValueError("collection light path scales require absolute prim paths and positive values")
    stage = Usd.Stage.Open(str(wrapper_path))
    if stage is None:
        raise RuntimeError(f"could not open run-local wrapper {wrapper_path}")
    stage.SetEditTarget(stage.GetRootLayer())
    deinstanced_roots: list[str] = []
    for _pass in range(3):
        proxy_roots: set[str] = set()
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            type_name = str(prim.GetTypeName())
            if not prim.IsInstanceProxy() or not (
                type_name.endswith("Light") or prim.HasAPI(UsdLux.LightAPI)
            ):
                continue
            cursor = prim
            while cursor and not cursor.IsInstance():
                cursor = cursor.GetParent()
            if cursor and cursor.IsInstance():
                proxy_roots.add(str(cursor.GetPath()))
        if not proxy_roots:
            break
        for path in sorted(proxy_roots):
            stage.GetPrimAtPath(path).SetInstanceable(False)
            if path not in deinstanced_roots:
                deinstanced_roots.append(path)
        stage.GetRootLayer().Save()
        stage = Usd.Stage.Open(str(wrapper_path))
        if stage is None:
            raise RuntimeError(f"could not reopen de-instanced wrapper {wrapper_path}")
        stage.SetEditTarget(stage.GetRootLayer())
    rows: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    remaining_proxy_lights = []
    matched_path_scales: set[str] = set()
    dome_background_override_count = 0
    dome_texture_override_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        type_name = str(prim.GetTypeName())
        if not (type_name.endswith("Light") or prim.HasAPI(UsdLux.LightAPI)):
            continue
        if prim.IsInstanceProxy():
            remaining_proxy_lights.append(str(prim.GetPath()))
            continue
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        intensity = UsdLux.LightAPI(prim).GetIntensityAttr()
        authored = intensity.Get()
        authored = 1.0 if authored is None else float(authored)
        applied_scale = float(scale)
        scale_group = "base"
        if type_name == "DomeLight" and dome_scale is not None:
            applied_scale = float(dome_scale)
            scale_group = "DomeLight"
        elif type_name == "DistantLight" and distant_scale is not None:
            applied_scale = float(distant_scale)
            scale_group = "DistantLight"
        elif type_name == "SphereLight":
            if sphere_scale is not None:
                applied_scale = float(sphere_scale)
                scale_group = "SphereLight"
            if strong_sphere_scale is not None and authored >= strong_sphere_intensity_threshold:
                applied_scale = float(strong_sphere_scale)
                scale_group = "strong_SphereLight"
        prim_path = str(prim.GetPath())
        if prim_path in path_scales:
            applied_scale = float(path_scales[prim_path])
            scale_group = f"path:{prim_path}"
            matched_path_scales.add(prim_path)
        scaled = authored * applied_scale
        intensity.Set(scaled)
        if type_name == "DomeLight":
            prim.CreateAttribute(
                "visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=False
            ).Set(bool(source_dome_background_visible))
            dome_background_override_count += 1
            texture_attr = prim.GetAttribute("inputs:texture:file")
            authored_texture = texture_attr.Get() if texture_attr else None
            if not source_dome_texture_enabled:
                if not texture_attr:
                    texture_attr = prim.CreateAttribute(
                        "inputs:texture:file", Sdf.ValueTypeNames.Asset, custom=False
                    )
                texture_attr.Set(Sdf.AssetPath(""))
                dome_texture_override_count += 1
        readback = float(intensity.Get())
        if not math.isclose(readback, scaled, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(f"light intensity readback mismatch at {prim.GetPath()}")
        rows.append(
            {
                "prim": str(prim.GetPath()),
                "type": type_name,
                "authored_intensity": authored,
                "applied_scale": applied_scale,
                "scale_group": scale_group,
                "scaled_intensity": readback,
                "authored_texture": (
                    str(authored_texture.path)
                    if type_name == "DomeLight" and authored_texture is not None
                    else None
                ),
            }
        )
    stage.GetRootLayer().Save()
    missing_path_scales = sorted(set(path_scales) - matched_path_scales)
    if missing_path_scales:
        raise RuntimeError("collection light path overrides matched no light: " + ", ".join(missing_path_scales))
    if remaining_proxy_lights:
        raise RuntimeError(
            f"{len(remaining_proxy_lights)} instance-proxy lights could not be overridden: "
            + ", ".join(remaining_proxy_lights[:4])
        )
    return {
        "source_usd_modified": False,
        "wrapper_path": str(wrapper_path),
        "scale": float(scale),
        "requested_scales": requested_scales,
        "strong_sphere_intensity_threshold": float(strong_sphere_intensity_threshold),
        "path_scales": path_scales,
        "source_dome_background_visible": bool(source_dome_background_visible),
        "source_dome_texture_enabled": bool(source_dome_texture_enabled),
        "dome_background_override_count": dome_background_override_count,
        "dome_texture_override_count": dome_texture_override_count,
        "type_counts": type_counts,
        "adjusted_count": len(rows),
        "deinstanced_roots": deinstanced_roots,
        "remaining_instance_proxy_light_count": len(remaining_proxy_lights),
        "adjusted": rows,
    }


def gpu_snapshot(index: int) -> dict[str, Any]:
    raw = run_text(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=index,name,uuid,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    fields = [field.strip() for field in raw.split(",")]
    keys = ["index", "name", "uuid", "driver", "memory_used_mib", "memory_total_mib", "utilization_percent", "temperature_c", "power_w"]
    return dict(zip(keys, fields)) if len(fields) == len(keys) else {"raw": raw}


def fixed_camera_matrix(name: str):
    from pxr import Gf

    views = {
        "fixed_kyoto01": ((-17.0, -7.0, 1.6), (-7.0, -7.0, 1.3)),
        "fixed_kyoto03": ((-7.0, -7.0, 1.6), (3.0, -7.0, 1.3)),
        "fixed_paris": ((-82.5, -4.0, 1.6), (-72.0, -4.0, 1.3)),
        "fixed_beijing": ((-36.0, 60.0, 1.6), (-26.0, 60.0, 1.3)),
    }
    eye, target = views[name]
    return Gf.Matrix4d(1.0).SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()


def source_camera_matrix(source_usd: Path, requested: str):
    from pxr import Usd, UsdGeom

    if requested.startswith("fixed_"):
        matrix = fixed_camera_matrix(requested)
        return matrix, {"method": "recorded_fixed_route_camera", "path": None}
    stage = Usd.Stage.Open(str(source_usd), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"could not open source stage for camera anchor: {source_usd}")
    prim = stage.GetPrimAtPath(requested)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"authored camera is not valid: {requested}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return matrix, {"method": "authored_camera", "path": requested}


def derive_route_anchor(matrix, route_policy: dict[str, Any]) -> dict[str, Any]:
    from pxr import Gf

    eye = matrix.ExtractTranslation()
    forward = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    offset = float(route_policy["forward_offset_from_recorded_camera_stage_units"])
    camera_height = float(route_policy["camera_height_above_route_ground_stage_units"])
    base_height = float(route_policy["go2_initial_base_height_stage_units"])
    centre = eye + forward * offset
    position = [float(centre[0]), float(centre[1]), float(eye[2]) - camera_height + base_height]
    yaw = math.atan2(float(forward[1]), float(forward[0]))
    return {
        "camera_eye": [float(eye[index]) for index in range(3)],
        "camera_forward": [float(forward[index]) for index in range(3)],
        "spawn_position": position,
        "spawn_yaw_rad": yaw,
        "estimated_ground_z": position[2] - base_height,
        "derivation": "camera eye + forward*offset for XY; camera_z-camera_height+Go2 base height for Z",
    }


def schedule_for_scene(
    config: dict[str, Any],
    turn_sign: float,
    duration_scale: float,
    target_duration_s: float | None = None,
) -> list[dict[str, Any]]:
    if duration_scale <= 0.0:
        raise ValueError("route duration scale must be positive")
    rows = []
    for item in config["default_schedule"]:
        row = dict(item)
        row["duration_s"] = float(row["duration_s"]) * duration_scale
        if row["label"] == "turn":
            row["angular_z"] = float(row["angular_z"]) * turn_sign
        rows.append(row)
    if target_duration_s is None:
        return rows
    if target_duration_s <= 0.0:
        raise ValueError("route duration seconds must be positive")
    # Repeating a short, already-smoked motion cycle is safer than stretching
    # one turn by 15-40x.  The repeated settle/stop segments also exercise the
    # requested stop/restart transition while keeping the route local.
    repeated: list[dict[str, Any]] = []
    elapsed = 0.0
    cycle = 0
    while elapsed < target_duration_s - 1e-9:
        cycle += 1
        for item in rows:
            remaining = target_duration_s - elapsed
            if remaining <= 1e-9:
                break
            row = dict(item)
            row["source_label"] = str(item["label"])
            row["label"] = f"cycle_{cycle:02d}/{item['label']}"
            row["duration_s"] = min(float(item["duration_s"]), remaining)
            repeated.append(row)
            elapsed += float(row["duration_s"])
    return repeated


def command_at_time(schedule: list[dict[str, Any]], timestamp: float) -> tuple[dict[str, Any], int]:
    cursor = 0.0
    for index, row in enumerate(schedule):
        end = cursor + float(row["duration_s"])
        if timestamp < end or index == len(schedule) - 1:
            return row, index
        cursor = end
    return schedule[-1], len(schedule) - 1


def integrate_reference(schedule: list[dict[str, Any]], start: list[float], yaw: float, dt: float) -> list[dict[str, Any]]:
    total_s = sum(float(row["duration_s"]) for row in schedule)
    count = int(round(total_s / dt))
    x, y, z, heading = float(start[0]), float(start[1]), float(start[2]), float(yaw)
    rows = [{"step_index": 0, "timestamp_s": 0.0, "position": [x, y, z], "yaw_rad": heading}]
    for step in range(count):
        timestamp = step * dt
        command, segment = command_at_time(schedule, timestamp)
        vx, vy, wz = float(command["linear_x"]), float(command["linear_y"]), float(command["angular_z"])
        x += (vx * math.cos(heading) - vy * math.sin(heading)) * dt
        y += (vx * math.sin(heading) + vy * math.cos(heading)) * dt
        heading += wz * dt
        rows.append(
            {
                "step_index": step + 1,
                "timestamp_s": (step + 1) * dt,
                "position": [x, y, z],
                "yaw_rad": heading,
                "schedule_index": segment,
                "schedule_label": command["label"],
                "command_body": [vx, vy, wz],
            }
        )
    return rows


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def coda_body_camera_transform(definition: dict[str, Any]) -> np.ndarray:
    """Return T_body_camera for the actual OpenCV optical output frame."""
    calibration = definition["calibration"]
    if calibration["role"] == "center_forward_pinhole" and definition.get("document_camera_ext"):
        return np.asarray(definition["document_camera_ext"]["T_vehicle_camera"], dtype=np.float64)
    yaw = float(definition["mount_yaw_rad"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    body_from_forward_frame = np.asarray(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    # OpenCV optical x/right, y/down, z/forward -> body-style x/forward,
    # y/left, z/up before the mount yaw is applied.
    forward_frame_from_optical = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = body_from_forward_frame @ forward_frame_from_optical
    transform[:3, 3] = np.asarray(definition["mount_position_vehicle_xyz_m"], dtype=np.float64)
    return transform


def yaw_from_quat(quat: np.ndarray) -> float:
    matrix = quat_wxyz_to_matrix(quat)
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def world_camera_matrix(base_position: np.ndarray, base_quat: np.ndarray, calibration: dict[str, Any]):
    from pxr import Gf

    rotation = quat_wxyz_to_matrix(base_quat)
    offset = np.asarray(calibration["position_vehicle_xyz_m"], dtype=np.float64)
    eye = base_position + rotation @ offset
    if "view_yaw_rad" in calibration:
        view_yaw = float(calibration["view_yaw_rad"])
        direction = rotation @ np.asarray([math.cos(view_yaw), math.sin(view_yaw), 0.0])
    elif calibration["role"] == "left_fisheye":
        direction = rotation @ np.asarray([0.0, 1.0, 0.0])
    elif calibration["role"] == "right_fisheye":
        direction = rotation @ np.asarray([0.0, -1.0, 0.0])
    else:
        direction = rotation @ np.asarray([1.0, 0.0, 0.0])
    up = rotation @ np.asarray([0.0, 0.0, 1.0])
    return Gf.Matrix4d(1.0).SetLookAt(
        Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*(eye + direction).tolist()), Gf.Vec3d(*up.tolist())
    ).GetInverse()


def overview_camera_matrix(base_position: np.ndarray, base_quat: np.ndarray):
    from pxr import Gf

    eye, target = overview_camera_view(base_position, base_quat)
    return overview_camera_matrix_from_view(eye, target)


def overview_camera_matrix_from_view(eye: np.ndarray, target: np.ndarray):
    from pxr import Gf

    return Gf.Matrix4d(1.0).SetLookAt(
        Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*target.tolist()), Gf.Vec3d(0.0, 0.0, 1.0)
    ).GetInverse()


def overview_camera_view(
    base_position: np.ndarray,
    base_quat: np.ndarray,
    chase_distance: float = 2.0,
    lateral_offset: float = 2.8,
    chase_height: float = 9.0,
    target_forward: float = 1.2,
    target_height: float = 0.25,
    target_lateral: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a wide elevated chase view for robot/vehicle interaction evidence."""
    yaw = yaw_from_quat(base_quat)
    forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.asarray([-forward[1], forward[0], 0.0])
    eye = (
        base_position
        - forward * chase_distance
        + left * lateral_offset
        + np.asarray([0.0, 0.0, chase_height])
    )
    target = (
        base_position
        + forward * target_forward
        + left * target_lateral
        + np.asarray([0.0, 0.0, target_height])
    )
    return eye, target


def phase1_overview_camera_view(
    route_start: np.ndarray, conflict_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed wide view: camera motion must not hide the vehicle's displacement."""
    focus_xy = (np.asarray(route_start[:2], dtype=np.float64) + conflict_xy) * 0.5
    target = np.asarray([focus_xy[0], focus_xy[1], 0.35], dtype=np.float64)
    eye = target + np.asarray([-4.0, -6.0, 10.0], dtype=np.float64)
    return eye, target


def probe_route_ground(anchor: dict[str, Any], query) -> dict[str, Any]:
    """Probe nearby source colliders without hitting the robot at the centre."""
    import carb

    x, y, z = [float(value) for value in anchor["spawn_position"]]
    yaw = float(anchor["spawn_yaw_rad"])
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    offsets = [forward * 0.75, -forward * 0.75, left * 0.75, -left * 0.75]
    rows = []
    for offset in offsets:
        origin = carb.Float3(float(x + offset[0]), float(y + offset[1]), float(z + 8.0))
        result = query.raycast_closest(origin, carb.Float3(0.0, 0.0, -1.0), 100.0)
        row: dict[str, Any] = {"origin": [float(origin[0]), float(origin[1]), float(origin[2])], "hit": False}
        if isinstance(result, dict) and bool(result.get("hit", False)):
            collision = str(result.get("collision") or "")
            rigid_body = str(result.get("rigidBody") or "")
            position = result.get("position")
            row.update(
                {
                    "hit": True,
                    "position": [float(position[index]) for index in range(3)] if position is not None else None,
                    "distance": float(result.get("distance")) if result.get("distance") is not None else None,
                    "collision": collision,
                    "rigid_body": rigid_body,
                    "robot_hit": "/Robot" in collision or "/Robot" in rigid_body,
                }
            )
        rows.append(row)
    usable = [row for row in rows if row.get("hit") and not row.get("robot_hit") and row.get("position")]
    clusters: list[list[dict[str, Any]]] = []
    for row in sorted(usable, key=lambda item: float(item["position"][2])):
        z = float(row["position"][2])
        if clusters and abs(z - float(np.median([item["position"][2] for item in clusters[-1]]))) <= 0.30:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    ranked = sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster),
            float(np.ptp([item["position"][2] for item in cluster])) if len(cluster) > 1 else 0.0,
            float(np.median([item["position"][2] for item in cluster])),
        ),
    )
    selected = ranked[0] if ranked else []
    selected_z = float(np.median([row["position"][2] for row in selected])) if selected else None
    selected_xy = (
        [float(np.mean([row["position"][axis] for row in selected])) for axis in (0, 1)] if selected else None
    )
    z_span = float(np.ptp([row["position"][2] for row in usable])) if len(usable) > 1 else 0.0
    safety_probe = None
    if selected_xy is not None and z_span > 0.50 and len(ranked) > 1:
        rejected = ranked[1]
        rejected_xy = np.asarray(
            [float(np.mean([row["position"][axis] for row in rejected])) for axis in (0, 1)],
            dtype=np.float64,
        )
        away = np.asarray(selected_xy, dtype=np.float64) - rejected_xy
        norm = float(np.linalg.norm(away))
        if norm > 1e-6:
            candidate_xy = np.asarray(selected_xy, dtype=np.float64) + away / norm * 1.25
            candidate_origin = carb.Float3(
                float(candidate_xy[0]),
                float(candidate_xy[1]),
                float(anchor["spawn_position"][2] + 8.0),
            )
            candidate_result = query.raycast_closest(
                candidate_origin, carb.Float3(0.0, 0.0, -1.0), 100.0
            )
            candidate_position = candidate_result.get("position") if isinstance(candidate_result, dict) else None
            candidate_z = float(candidate_position[2]) if candidate_position is not None else None
            accepted = bool(
                isinstance(candidate_result, dict)
                and candidate_result.get("hit")
                and candidate_z is not None
                and selected_z is not None
                and abs(candidate_z - selected_z) <= 0.30
                and "/Robot" not in str(candidate_result.get("collision") or "")
            )
            safety_probe = {
                "reason": "move away from the nearest rejected higher support cluster",
                "clearance_offset_stage_units": 1.25,
                "candidate_xy": candidate_xy.tolist(),
                "hit": bool(isinstance(candidate_result, dict) and candidate_result.get("hit")),
                "position": [float(candidate_position[index]) for index in range(3)]
                if candidate_position is not None
                else None,
                "collision": str(candidate_result.get("collision") or "")
                if isinstance(candidate_result, dict)
                else None,
                "accepted": accepted,
            }
            if accepted:
                selected_xy = candidate_xy.tolist()
    return {
        "direction": [0.0, 0.0, -1.0],
        "probe_radius_stage_units": 0.75,
        "samples": rows,
        "usable_hit_count": len(usable),
        "height_cluster_tolerance_stage_units": 0.30,
        "height_clusters": [
            {
                "sample_count": len(cluster),
                "median_z": float(np.median([row["position"][2] for row in cluster])),
                "z_span": float(np.ptp([row["position"][2] for row in cluster])) if len(cluster) > 1 else 0.0,
                "sample_collisions": sorted({str(row.get("collision") or "") for row in cluster}),
            }
            for cluster in ranked
        ],
        "all_usable_z_span": z_span,
        "selected_ground_z": selected_z,
        "selected_ground_xy": selected_xy,
        "selected_ground_xy_safety_probe": safety_probe,
        "selected_cluster_sample_count": len(selected),
        "selection": "cluster nearby non-robot hits by Z; prefer largest, then flattest, then lowest support cluster",
    }


def probe_safe_road_corridor(
    origin_xy: np.ndarray,
    ground_z: float,
    base_yaw_rad: float,
    query,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Choose a flat, obstacle-free out-and-back corridor before Go2 moves.

    This is deterministic route admission, not autonomous obstacle avoidance.
    Downward rays verify continuous support on a road-like collider; horizontal
    rays reject visible or invisible blocking geometry in the Go2 envelope.
    """
    import carb

    route_extent = float(config.get("route_extent_stage_units", 2.5))
    clearance_margin = float(config.get("clearance_margin_stage_units", 0.5))
    required_length = route_extent + clearance_margin
    sample_step = float(config.get("sample_step_stage_units", 0.25))
    sample_start = float(config.get("sample_start_stage_units", 0.5))
    lateral_offsets = [float(value) for value in config.get("lateral_offsets_stage_units", [-0.35, 0.0, 0.35])]
    height_tolerance = float(config.get("ground_height_tolerance_stage_units", 0.15))
    obstacle_heights = [float(value) for value in config.get("obstacle_heights_stage_units", [0.18, 0.40, 0.65])]
    allowed_tokens = [str(value).lower() for value in config.get("allowed_ground_path_tokens", [])]
    rejected_tokens = [str(value).lower() for value in config.get("rejected_ground_path_tokens", [])]
    heading_offsets = [float(value) for value in config.get("candidate_heading_offsets_deg", [-90, 90, 0, 180])]
    distance_count = max(1, int(math.ceil((required_length - sample_start) / sample_step)) + 1)
    distances = [min(required_length, sample_start + index * sample_step) for index in range(distance_count)]
    distances = sorted(set(round(value, 6) for value in distances))

    def raycast(origin_xyz: list[float], direction_xyz: list[float], max_distance: float) -> dict[str, Any]:
        result = query.raycast_closest(carb.Float3(*origin_xyz), carb.Float3(*direction_xyz), max_distance)
        row: dict[str, Any] = {
            "origin": origin_xyz,
            "direction": direction_xyz,
            "max_distance": float(max_distance),
            "hit": False,
        }
        if isinstance(result, dict) and bool(result.get("hit", False)):
            position = result.get("position")
            collision = str(result.get("collision") or "")
            rigid_body = str(result.get("rigidBody") or "")
            row.update(
                {
                    "hit": True,
                    "position": [float(position[index]) for index in range(3)] if position is not None else None,
                    "distance": float(result.get("distance")) if result.get("distance") is not None else None,
                    "collision": collision,
                    "rigid_body": rigid_body,
                    "robot_hit": "/Robot" in collision or "/Robot" in rigid_body,
                }
            )
        return row

    candidates: list[dict[str, Any]] = []
    for order, offset_deg in enumerate(heading_offsets):
        yaw = navigation.wrap_angle(base_yaw_rad + math.radians(offset_deg))
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        ground_samples: list[dict[str, Any]] = []
        accepted_by_distance: dict[float, bool] = {}
        for distance in distances:
            distance_ok = True
            for lateral in lateral_offsets:
                xy = origin_xy + forward * distance + left * lateral
                row = raycast(
                    [float(xy[0]), float(xy[1]), float(ground_z + 3.0)],
                    [0.0, 0.0, -1.0],
                    6.0,
                )
                path_text = f"{row.get('collision', '')} {row.get('rigid_body', '')}".lower()
                position = row.get("position")
                height_ok = bool(position is not None and abs(float(position[2]) - ground_z) <= height_tolerance)
                token_ok = bool(not allowed_tokens or any(token in path_text for token in allowed_tokens))
                rejected = any(token in path_text for token in rejected_tokens)
                accepted = bool(row.get("hit") and not row.get("robot_hit") and height_ok and token_ok and not rejected)
                row.update(
                    {
                        "route_distance_stage_units": distance,
                        "lateral_offset_stage_units": lateral,
                        "ground_height_delta_stage_units": (
                            float(position[2]) - ground_z if position is not None else None
                        ),
                        "road_path_token_matched": token_ok,
                        "rejected_path_token_matched": rejected,
                        "accepted": accepted,
                    }
                )
                ground_samples.append(row)
                distance_ok = distance_ok and accepted
            accepted_by_distance[distance] = distance_ok

        clear_distance = 0.0
        for distance in distances:
            if not accepted_by_distance[distance]:
                break
            clear_distance = distance

        obstacle_rays: list[dict[str, Any]] = []
        horizontal_start = sample_start
        horizontal_length = max(0.01, required_length - horizontal_start)
        for lateral in lateral_offsets:
            for height in obstacle_heights:
                xy = origin_xy + forward * horizontal_start + left * lateral
                row = raycast(
                    [float(xy[0]), float(xy[1]), float(ground_z + height)],
                    [float(forward[0]), float(forward[1]), 0.0],
                    horizontal_length,
                )
                blocked = bool(row.get("hit") and not row.get("robot_hit"))
                row.update(
                    {
                        "lateral_offset_stage_units": lateral,
                        "height_above_ground_stage_units": height,
                        "blocked": blocked,
                    }
                )
                obstacle_rays.append(row)

        endpoint_xy = origin_xy + forward * route_extent
        endpoint_clearance_rays: list[dict[str, Any]] = []
        endpoint_radius = float(config.get("endpoint_clearance_radius_stage_units", 0.55))
        for angle_deg in range(0, 360, 45):
            angle = math.radians(angle_deg)
            direction = [math.cos(angle), math.sin(angle), 0.0]
            for height in obstacle_heights:
                row = raycast(
                    [float(endpoint_xy[0]), float(endpoint_xy[1]), float(ground_z + height)],
                    direction,
                    endpoint_radius,
                )
                blocked = bool(row.get("hit") and not row.get("robot_hit"))
                row.update(
                    {
                        "angle_deg": angle_deg,
                        "height_above_ground_stage_units": height,
                        "blocked": blocked,
                    }
                )
                endpoint_clearance_rays.append(row)

        blocked_corridor = [row for row in obstacle_rays if row["blocked"]]
        blocked_endpoint = [row for row in endpoint_clearance_rays if row["blocked"]]
        ground_accepted_count = sum(bool(row["accepted"]) for row in ground_samples)
        accepted = bool(
            clear_distance >= required_length - 1.0e-6
            and not blocked_corridor
            and not blocked_endpoint
        )
        candidates.append(
            {
                "order": order,
                "heading_offset_deg": offset_deg,
                "world_yaw_rad": yaw,
                "forward_world_xy": forward.tolist(),
                "route_endpoint_world_xy": endpoint_xy.tolist(),
                "accepted": accepted,
                "continuous_clear_distance_stage_units": clear_distance,
                "ground_accepted_count": ground_accepted_count,
                "ground_sample_count": len(ground_samples),
                "blocked_corridor_ray_count": len(blocked_corridor),
                "blocked_endpoint_ray_count": len(blocked_endpoint),
                "ground_samples": ground_samples,
                "obstacle_rays": obstacle_rays,
                "endpoint_clearance_rays": endpoint_clearance_rays,
            }
        )

    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    selected = max(
        accepted_candidates,
        key=lambda candidate: (
            candidate["continuous_clear_distance_stage_units"],
            candidate["ground_accepted_count"],
            -abs(candidate["heading_offset_deg"]),
            -candidate["order"],
        ),
        default=None,
    )
    return {
        "enabled": True,
        "purpose": "deterministic road-corridor admission before motion; not autonomous obstacle avoidance",
        "origin_world_xy": origin_xy.tolist(),
        "ground_z_stage_units": float(ground_z),
        "base_yaw_rad": float(base_yaw_rad),
        "route_extent_stage_units": route_extent,
        "clearance_margin_stage_units": clearance_margin,
        "required_clear_distance_stage_units": required_length,
        "sample_step_stage_units": sample_step,
        "lateral_offsets_stage_units": lateral_offsets,
        "ground_height_tolerance_stage_units": height_tolerance,
        "allowed_ground_path_tokens": allowed_tokens,
        "rejected_ground_path_tokens": rejected_tokens,
        "candidates": candidates,
        "accepted_candidate_count": len(accepted_candidates),
        "selected_heading_offset_deg": selected["heading_offset_deg"] if selected else None,
        "selected_world_yaw_rad": selected["world_yaw_rad"] if selected else None,
        "selected_endpoint_world_xy": selected["route_endpoint_world_xy"] if selected else None,
        "selection": "prefer full clear distance, then ground coverage, then the smallest initial heading change",
        "accepted": selected is not None,
    }


def build_safe_shuttle_plan(
    origin_xy: np.ndarray, road_probe: dict[str, Any], allow_secondary_spoke: bool = False
) -> dict[str, Any]:
    """Build forward-only out-and-back legs from accepted road-spoke evidence."""
    if not road_probe.get("accepted"):
        raise ValueError("safe shuttle requires an accepted road corridor")
    selected_offset = float(road_probe["selected_heading_offset_deg"])
    selected = next(
        candidate
        for candidate in road_probe["candidates"]
        if candidate["accepted"] and math.isclose(float(candidate["heading_offset_deg"]), selected_offset)
    )
    secondary_options = []
    for candidate in road_probe["candidates"]:
        if not candidate["accepted"] or candidate is selected:
            continue
        separation = abs(
            navigation.wrap_angle(float(candidate["world_yaw_rad"]) - float(selected["world_yaw_rad"]))
        )
        if math.radians(15.0) <= separation <= math.radians(75.0):
            secondary_options.append((separation, int(candidate["order"]), candidate))
    secondary = min(secondary_options, default=(None, None, None))[2] if allow_secondary_spoke else None

    origin = np.asarray(origin_xy, dtype=np.float64).tolist()
    spokes = [("primary", selected)]
    if secondary is not None:
        spokes.append(("secondary", secondary))
    legs = []
    for name, spoke in spokes:
        endpoint = list(spoke["route_endpoint_world_xy"])
        yaw = float(spoke["world_yaw_rad"])
        legs.extend(
            [
                {
                    "start_world_xy": origin,
                    "target_world_xy": endpoint,
                    "body_yaw_rad": yaw,
                    "reverse": False,
                    "label": f"{name}_forward",
                    "source_heading_offset_deg": float(spoke["heading_offset_deg"]),
                },
                {
                    "start_world_xy": endpoint,
                    "target_world_xy": origin,
                    "body_yaw_rad": navigation.wrap_angle(yaw + math.pi),
                    "reverse": False,
                    "label": f"{name}_turnaround_return",
                    "source_heading_offset_deg": float(spoke["heading_offset_deg"]),
                },
            ]
        )
    return {
        "mode": "prevalidated_forward_turnaround_road_shuttle",
        "purpose": "long-running low-difficulty forward capture inside prevalidated road corridors",
        "legs": legs,
        "secondary_spoke_used": secondary is not None,
        "secondary_spoke_allowed": bool(allow_secondary_spoke),
        "secondary_heading_offset_deg": float(secondary["heading_offset_deg"]) if secondary is not None else None,
        "limitations": (
            "Endpoint turns use the frozen low-level policy and are monitored, but they do not prove autonomous "
            "obstacle avoidance. A secondary spoke is used only when its complete corridor passed the same PhysX "
            "admission test."
        ),
    }


def probe_right_angle_road_route(
    origin_xy: np.ndarray,
    ground_z: float,
    base_yaw_rad: float,
    query,
    config: dict[str, Any],
    initial_probe: dict[str, Any],
    corner_distances: list[float],
    second_leg_extent: float,
    requested_turn_deg: float | None = None,
) -> dict[str, Any]:
    """Admit an L route only when both road legs and the turn area are clear."""
    if not initial_probe.get("accepted"):
        raise ValueError("right-angle admission requires an accepted initial road corridor")
    primary_yaw = float(initial_probe["selected_world_yaw_rad"])
    primary_offset = float(initial_probe["selected_heading_offset_deg"])
    primary_forward = np.asarray([math.cos(primary_yaw), math.sin(primary_yaw)], dtype=np.float64)
    maximum_first_extent = float(initial_probe["route_extent_stage_units"])
    turn_candidates = (
        [float(requested_turn_deg)]
        if requested_turn_deg is not None
        else [-90.0, 90.0]
    )
    rows: list[dict[str, Any]] = []
    for distance in sorted(set(float(value) for value in corner_distances)):
        row: dict[str, Any] = {
            "first_leg_extent_stage_units": distance,
            "accepted": False,
        }
        if distance <= 0.5 or distance > maximum_first_extent:
            row["rejection_reason"] = "corner distance is outside the admitted initial corridor"
            rows.append(row)
            continue
        first_config = copy.deepcopy(config)
        first_config["route_extent_stage_units"] = distance
        first_config["candidate_heading_offsets_deg"] = [primary_offset]
        first_probe = probe_safe_road_corridor(
            origin_xy, ground_z, base_yaw_rad, query, first_config
        )
        corner_xy = np.asarray(origin_xy, dtype=np.float64) + primary_forward * distance
        second_config = copy.deepcopy(config)
        second_config["route_extent_stage_units"] = float(second_leg_extent)
        second_config["candidate_heading_offsets_deg"] = turn_candidates
        second_probe = probe_safe_road_corridor(
            corner_xy, ground_z, primary_yaw, query, second_config
        )
        accepted = bool(first_probe.get("accepted") and second_probe.get("accepted"))
        final_xy = second_probe.get("selected_endpoint_world_xy") if accepted else None
        turn_angle = second_probe.get("selected_heading_offset_deg") if accepted else None
        row.update(
            {
                "corner_world_xy": corner_xy.tolist(),
                "second_leg_extent_stage_units": float(second_leg_extent),
                "first_leg_probe": first_probe,
                "second_leg_probe": second_probe,
                "turn_angle_deg": turn_angle,
                "final_world_xy": final_xy,
                "total_route_length_stage_units": distance + float(second_leg_extent),
                "accepted": accepted,
                "rejection_reason": None if accepted else "first leg, turn clearance or second leg was rejected",
            }
        )
        rows.append(row)
    accepted_rows = [row for row in rows if row["accepted"]]
    selected = max(
        accepted_rows,
        key=lambda row: (
            float(row["total_route_length_stage_units"]),
            float(row["first_leg_extent_stage_units"]),
            -abs(float(row["turn_angle_deg"])),
        ),
        default=None,
    )
    return {
        "enabled": True,
        "purpose": "prevalidate a road-aligned L route before physical Go2 motion",
        "origin_world_xy": np.asarray(origin_xy, dtype=np.float64).tolist(),
        "initial_heading_world_rad": primary_yaw,
        "requested_turn_candidates_deg": turn_candidates,
        "requested_corner_distances_stage_units": sorted(
            set(float(value) for value in corner_distances)
        ),
        "requested_second_leg_extent_stage_units": float(second_leg_extent),
        "candidates": rows,
        "accepted_candidate_count": len(accepted_rows),
        "accepted": selected is not None,
        "selected": selected,
        "waypoints_world_xy": (
            [
                np.asarray(origin_xy, dtype=np.float64).tolist(),
                list(selected["corner_world_xy"]),
                list(selected["final_world_xy"]),
            ]
            if selected is not None
            else None
        ),
        "limitations": (
            "This is deterministic geometric admission plus waypoint following, not autonomous route planning. "
            "The frozen policy still has to execute the turn and is monitored for stuck/non-foot contact."
        ),
    }


def matrix_rows(matrix) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def normalise_rgb(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise RuntimeError(f"unexpected RGB shape {array.shape}")
    return np.clip(array[..., :3], 0, 255).astype(np.uint8)


def normalise_array(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).squeeze()


def route_camera_application(profile: str, seed: int) -> dict[str, Any] | None:
    """Return the deterministic enhanced camera profile before scene creation."""
    if profile != "camera_wide_bright":
        return None
    variant = next(item for item in appearance.VARIANTS if item["id"] == profile)
    rng = np.random.default_rng(seed + int(variant["seed_offset"]))
    translation = [float(value) for value in rng.uniform([0.24, -0.32, 0.10], [0.42, -0.16, 0.22])]
    yaw_deg = float(rng.uniform(-9.0, -6.0))
    focal_scale = float(rng.uniform(0.58, 0.70))
    aperture_scale = float(rng.uniform(1.12, 1.24))
    exposure = float(rng.uniform(1.05, 1.35))
    return {
        "id": profile,
        "kind": "camera",
        "profile": "wide_bright",
        "tests": list(variant["tests"]),
        "seed": seed + int(variant["seed_offset"]),
        "proxy": False,
        "application": "Isaac Lab CameraCfg mount/lens override authored before Fabric initialization",
        "translation_noise_vehicle_m": translation,
        "translation_noise_m": translation,
        "rotation_noise_yaw_deg": yaw_deg,
        "focal_length_scale": focal_scale,
        "aperture_scale": aperture_scale,
        "exposure_ev": exposure,
        "display_name_zh": variant.get("display_name_zh", profile),
        "explanation_zh": variant.get("explanation_zh", ""),
    }


def coda_pinhole_calibrations(
    width: int, height: int, horizontal_fov_deg: float, side_yaw_deg: float
) -> list[dict[str, Any]]:
    """Build the actual-output K/D and mount definitions for a three-pinhole rig.

    Body yaw is positive to the left.  At the default 80 degree horizontal
    FOV and 50 degree adjacent yaw separation, neighbouring ideal pinhole
    frusta overlap by approximately 37.5% of their horizontal angular span.
    """
    if width < 960 or height < 640:
        raise ValueError("CoDa pinhole rig requires at least 960x640 output")
    if not (80.0 <= horizontal_fov_deg <= 100.0):
        raise ValueError("CoDa pinhole horizontal FOV must be within 80-100 degrees")
    overlap_ratio = (horizontal_fov_deg - side_yaw_deg) / horizontal_fov_deg
    if not (0.20 <= overlap_ratio <= 0.40):
        raise ValueError(
            f"adjacent pinhole overlap must be 20%-40%, got {overlap_ratio:.3f}"
        )
    fx = 0.5 * float(width) / math.tan(0.5 * math.radians(horizontal_fov_deg))
    fy = fx
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    shared = {
        "requested_projection_type": "pinhole",
        "native_projection_type": "pinhole",
        "source_document_model": "generated ideal pinhole",
        "camera_model": "pinhole",
        "distortion_model": "opencv_radtan",
        "K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "D": [0.0, 0.0, 0.0, 0.0],
        "image_size": [int(width), int(height)],
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "adjacent_overlap_angular_ratio": float(overlap_ratio),
    }
    return [
        {
            **shared,
            "id": "camera_coda_front_pinhole",
            "role": "center_forward_pinhole",
            "delivery_name": "cam_front",
            "position_vehicle_xyz_m": [0.55, 0.0, 0.55],
            "view_yaw_rad": 0.0,
        },
        {
            **shared,
            "id": "camera_coda_front_left_pinhole",
            "role": "left_forward_pinhole",
            "delivery_name": "cam_front_left",
            "position_vehicle_xyz_m": [0.55, 0.12, 0.55],
            "view_yaw_rad": math.radians(float(side_yaw_deg)),
        },
        {
            **shared,
            "id": "camera_coda_front_right_pinhole",
            "role": "right_forward_pinhole",
            "delivery_name": "cam_front_right",
            "position_vehicle_xyz_m": [0.55, -0.12, 0.55],
            "view_yaw_rad": -math.radians(float(side_yaw_deg)),
        },
    ]


def add_navigation_camera_sensors(
    cfg,
    args: argparse.Namespace,
    camera_application: dict[str, Any] | None = None,
    camera_calibrations: list[dict[str, Any]] | None = None,
    onboard_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Add body-mounted Isaac Lab cameras before Fabric is initialized.

    Creating Replicator camera Xforms after the task has initialized Fabric can
    produce a 90-degree optical-roll mismatch on Isaac Sim 4.5.  Isaac Lab's
    CameraCfg performs the world/ROS/OpenGL conversion before scene start and
    keeps the cameras attached to the articulated Go2 base.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    definitions: dict[str, dict[str, Any]] = {}
    # Keep the controller, pose and contact streams at the native physics rate,
    # but refresh the high-resolution cameras only at the requested dataset
    # rate.  Each saved frame is force-recomputed for every camera on the same
    # simulation step below, so this removes discarded 50 Hz renders without
    # weakening cross-camera/timestamp alignment.
    camera_update_period_s = 1.0 / max(float(args.capture_fps), 1.0e-6)
    camera_calibrations = camera_calibrations or CAMERA_CALIBRATIONS
    selected_calibrations = (
        [cal for cal in camera_calibrations if cal["role"] == "center_forward_pinhole"]
        if args.lightweight_three_panel
        else camera_calibrations
    )
    for calibration in selected_calibrations:
        render_cal = calibration_for_resolution(calibration, args.sensor_width, args.sensor_height)
        if args.strict_document_calibration:
            requested = [args.sensor_width, args.sensor_height]
            documented = calibration["image_size"]
            exact = requested == documented
            same_aspect = math.isclose(
                float(args.sensor_width) / float(args.sensor_height),
                float(documented[0]) / float(documented[1]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            if not exact and not (args.allow_scaled_document_calibration and same_aspect):
                qualifier = " or an explicitly allowed same-aspect-ratio scaled resolution"
                raise RuntimeError(
                    f"strict document calibration requires {documented}{qualifier}, got {requested}"
                )
        intrinsics = camera_intrinsics(render_cal, 20.955, args.sensor_width, args.sensor_height)
        if args.strict_document_calibration:
            fx, fy = float(render_cal["K"][0]), float(render_cal["K"][4])
            strict_focal_mm = fx * 20.955 / float(args.sensor_width)
            intrinsics.update(
                {
                    "focal_length_mm": strict_focal_mm,
                    "horizontal_aperture_mm": 20.955,
                    "vertical_aperture_mm": strict_focal_mm * float(args.sensor_height) / fy,
                }
            )
        focal_scale = float((camera_application or {}).get("focal_length_scale", 1.0))
        aperture_scale = float((camera_application or {}).get("aperture_scale", 1.0))
        spawn_kwargs = {
            "projection_type": calibration["native_projection_type"],
            "focal_length": float(intrinsics["focal_length_mm"]) * focal_scale,
            "horizontal_aperture": float(intrinsics["horizontal_aperture_mm"]) * aperture_scale,
            "vertical_aperture": float(intrinsics["vertical_aperture_mm"]) * aperture_scale,
            "horizontal_aperture_offset": (float(intrinsics["cx_px"]) - args.sensor_width * 0.5)
            / args.sensor_width
            * float(intrinsics["horizontal_aperture_mm"])
            * aperture_scale,
            "vertical_aperture_offset": (args.sensor_height * 0.5 - float(intrinsics["cy_px"]))
            / args.sensor_height
            * float(intrinsics["vertical_aperture_mm"])
            * aperture_scale,
            "clipping_range": (0.02, 1000.0),
            "focus_distance": 10.0,
            "f_stop": 0.0,
        }
        if calibration["native_projection_type"] == "pinhole":
            spawn = sim_utils.PinholeCameraCfg(**spawn_kwargs)
        else:
            spawn_kwargs.update(
                {
                    "fisheye_nominal_width": float(args.sensor_width),
                    "fisheye_nominal_height": float(args.sensor_height),
                    "fisheye_optical_centre_x": float(intrinsics["cx_px"]),
                    "fisheye_optical_centre_y": float(intrinsics["cy_px"]),
                    "fisheye_max_fov": 190.0,
                    "fisheye_polynomial_a": 0.0,
                    "fisheye_polynomial_b": 1.0 / max(float(intrinsics["fx_px"]), 1.0),
                }
            )
            spawn = sim_utils.FisheyeCameraCfg(**spawn_kwargs)
        yaw = float(calibration.get("view_yaw_rad", 0.0))
        if calibration["role"] == "left_fisheye":
            yaw = math.pi * 0.5
        elif calibration["role"] == "right_fisheye":
            yaw = -math.pi * 0.5
        yaw += math.radians(float((camera_application or {}).get("rotation_noise_yaw_deg", 0.0)))
        if args.strict_document_calibration:
            if camera_application:
                raise RuntimeError("strict document calibration cannot be combined with camera randomization")
            ext = strict_calibration.camera_ext(calibration)
            if calibration["role"] == "center_forward_pinhole":
                mount_quaternion = tuple(ext["quaternion_wxyz_vehicle_camera"])
                mount_convention = "ros"
                orientation_application = "document ROS optical CameraExt"
            else:
                # A ROS optical camera looks along +Z.  The supplied side RPY
                # [0, 0, +/-pi/2] only spins the image plane around +Z and
                # therefore leaves the optical axis pointing upward.  Resolve
                # those inconsistent blocks as horizontal outward body-yaw
                # mounts.  CameraExt is camera->vehicle, hence the sign flip.
                side_yaw = -float(calibration["rpy_vehicle_camera_rad"][2])
                mount_quaternion = (math.cos(side_yaw * 0.5), 0.0, 0.0, math.sin(side_yaw * 0.5))
                mount_convention = "world"
                orientation_application = (
                    "horizontal outward-facing vehicle/body-yaw interpretation; "
                    "document side ROS-optical RPY is internally inconsistent"
                )
        else:
            ext = None
            mount_quaternion = (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))
            mount_convention = "world"
            orientation_application = "explicit horizontal body-yaw ideal-pinhole layout"
        mount_position = np.asarray(calibration["position_vehicle_xyz_m"], dtype=np.float64)
        mount_position += np.asarray(
            (camera_application or {}).get("translation_noise_vehicle_m", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        key = f"go2_{calibration['id']}"
        data_types = ["distance_to_camera"] if args.lightweight_three_panel else ["rgb", "distance_to_camera"]
        if calibration["role"] == "center_forward_pinhole" and not args.omit_motion_vectors:
            data_types.append("motion_vectors")
        setattr(
            cfg.scene,
            key,
            CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/base/{calibration['id']}",
                update_period=camera_update_period_s,
                height=args.sensor_height,
                width=args.sensor_width,
                data_types=data_types,
                spawn=spawn,
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(value) for value in mount_position),
                    rot=mount_quaternion,
                    convention=mount_convention,
                ),
                update_latest_camera_pose=True,
            ),
        )
        definitions[calibration["id"]] = {
            "scene_key": key,
            "calibration": calibration,
            "render_calibration": render_cal,
            "mount_yaw_rad": yaw,
            "mount_position_vehicle_xyz_m": mount_position.tolist(),
            "mount_quaternion_wxyz": list(mount_quaternion),
            "mount_convention": mount_convention,
            "orientation_application": orientation_application,
            "update_period_s": camera_update_period_s,
            "document_camera_ext": ext,
        }

    if onboard_only:
        return definitions

    overview_spawn = sim_utils.PinholeCameraCfg(
        focal_length=10.0 if args.phase1_dynamic_vehicle else args.overview_focal_length,
        horizontal_aperture=20.955,
        clipping_range=(0.02, 1000.0),
        focus_distance=10.0,
        f_stop=0.0,
    )
    cfg.scene.go2_overview = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        update_period=camera_update_period_s,
        height=args.overview_height,
        width=args.overview_width,
        data_types=["rgb"] if args.lightweight_three_panel else ["rgb", "distance_to_camera"],
        spawn=overview_spawn,
        offset=CameraCfg.OffsetCfg(convention="world"),
        update_latest_camera_pose=True,
    )
    definitions["overview"] = {
        "scene_key": "go2_overview",
        "calibration": None,
        "render_calibration": None,
        "update_period_s": camera_update_period_s,
    }
    return definitions


def route_inset(draw: ImageDraw.ImageDraw, actual_xy: list[list[float]], reference_xy: list[list[float]], box) -> None:
    x0, y0, x1, y1 = box
    points = np.asarray(reference_xy + actual_xy, dtype=np.float64)
    if not len(points):
        return
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 0.5)

    def map_point(point):
        nx, ny = (np.asarray(point) - minimum) / span
        return (x0 + 12 + float(nx) * (x1 - x0 - 24), y1 - 12 - float(ny) * (y1 - y0 - 24))

    draw.rounded_rectangle(box, radius=8, fill=(4, 12, 18, 205), outline=(210, 230, 238, 180), width=2)
    if len(reference_xy) >= 2:
        draw.line([map_point(point) for point in reference_xy], fill=(60, 220, 120, 240), width=4)
    if len(actual_xy) >= 2:
        draw.line([map_point(point) for point in actual_xy], fill=(255, 170, 45, 255), width=4)
    if actual_xy:
        px, py = map_point(actual_xy[-1])
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 80, 75, 255))


def annotate_overview(
    rgb: np.ndarray,
    scene: str,
    frame_index: int,
    timestamp: float,
    command: dict[str, Any],
    position: list[float],
    speed: float,
    collision: bool,
    stuck: bool,
    actual_xy: list[list[float]],
    reference_xy: list[list[float]],
    appearance_profile: str,
    execution_mode: str,
    contact_evidence_valid: bool,
) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    # Keep diagnostics in a shallow top-left strip.  The dog and crossing
    # vehicle occupy the lower/central image and must remain unobstructed.
    title_font = load_font(max(13, width // 82), bold=True)
    body_font = load_font(max(10, width // 108))
    panel_left = 10
    panel_top = 10
    panel_width = min(width - 20, max(300, int(width * 0.54)))
    panel_height = 62
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_left + panel_width, panel_top + panel_height),
        radius=7,
        fill=(4, 12, 18, 170),
    )
    draw.text((panel_left + 10, panel_top + 5), "Go2 动态车辆让行", fill=(255, 255, 255, 255), font=title_font)
    lines = [
        f"t={timestamp:05.2f}s  {command['label']}  speed={speed:.2f} m/s",
        f"cmd=({command['linear_x']:.2f}, {command['linear_y']:.2f}, {command['angular_z']:.2f})  "
        f"collision={'YES' if collision else 'no'}  stuck={'YES' if stuck else 'no'}",
    ]
    for index, line in enumerate(lines):
        draw.text(
            (panel_left + 10, panel_top + 25 + index * 16),
            line,
            fill=(225, 239, 244, 255),
            font=body_font,
        )
    inset_width = min(190, max(110, rgb.shape[1] // 5))
    route_inset(
        draw,
        actual_xy,
        reference_xy,
        (rgb.shape[1] - inset_width - 18, 18, rgb.shape[1] - 18, min(rgb.shape[0] - 18, 18 + inset_width)),
    )
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def encode_video(frames: list[np.ndarray], output: Path, fps: float) -> dict[str, Any]:
    import cv2

    if not frames:
        raise ValueError("no overview frames")
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"VP90"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not open VP9 writer")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    reader = cv2.VideoCapture(str(output))
    metadata = {
        "opened": bool(reader.isOpened()),
        "frame_count": int(round(reader.get(cv2.CAP_PROP_FRAME_COUNT))),
        "width": int(round(reader.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(reader.get(cv2.CAP_PROP_FPS)),
    }
    reader.release()
    metadata["file_size_bytes"] = output.stat().st_size
    metadata["duration_s"] = metadata["frame_count"] / max(metadata["fps"], 1e-6)
    if not metadata["opened"] or metadata["frame_count"] != len(frames):
        raise RuntimeError(f"video readback failed: {metadata}, expected={len(frames)}")
    return metadata


def encode_video_paths(paths: list[Path], output: Path, fps: float) -> dict[str, Any]:
    """Encode saved review frames without retaining a long run in RAM."""
    if not paths:
        raise ValueError("no review frames")
    frames = (np.asarray(Image.open(path).convert("RGB")) for path in paths)
    first = next(frames)
    import cv2

    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"VP90"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not open VP9 writer")
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    reader = cv2.VideoCapture(str(output))
    metadata = {
        "opened": bool(reader.isOpened()),
        "frame_count": int(round(reader.get(cv2.CAP_PROP_FRAME_COUNT))),
        "width": int(round(reader.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(reader.get(cv2.CAP_PROP_FPS)),
    }
    reader.release()
    metadata["file_size_bytes"] = output.stat().st_size
    metadata["duration_s"] = metadata["frame_count"] / max(metadata["fps"], 1e-6)
    if not metadata["opened"] or metadata["frame_count"] != len(paths):
        raise RuntimeError(f"video readback failed: {metadata}, expected={len(paths)}")
    return metadata


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, (3, 9, 14))
    result.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return result


def lightweight_review_frame(
    depth: np.ndarray,
    overview_rgb: np.ndarray,
    reference_xy: list[list[float]],
    actual_xy: list[list[float]],
    timestamp_s: float,
    capture_fps: float,
    playback_speed: float,
) -> np.ndarray:
    """Build the requested single-row trajectory/depth/third-person layout."""
    import cv2

    panel_w, panel_h, header_h = 480, 400, 42
    canvas = Image.new("RGB", (panel_w * 3, panel_h + header_h), (5, 12, 17))
    draw = ImageDraw.Draw(canvas)
    labels = ("参考轨迹 / 实际物理轨迹", "中央针孔 depth", "第三人称视角")
    font = load_font(20, bold=True)
    small = load_font(14)
    for index, label in enumerate(labels):
        draw.text((index * panel_w + 12, 9), label, fill=(235, 242, 245), font=font)
    route = Image.new("RGB", (panel_w, panel_h), (5, 14, 20))
    route_draw = ImageDraw.Draw(route, "RGBA")
    route_inset(route_draw, actual_xy, reference_xy, (10, 10, panel_w - 10, panel_h - 10))
    finite = np.isfinite(depth) & (depth > 0.0)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        values = depth[finite]
        lo, hi = np.percentile(values, [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1.0
        scaled[finite] = np.clip((depth[finite] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    depth_rgb[~finite] = 0
    panels = (
        route,
        fit_panel(Image.fromarray(depth_rgb, mode="RGB"), (panel_w, panel_h)),
        fit_panel(Image.fromarray(overview_rgb, mode="RGB"), (panel_w, panel_h)),
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel_w, header_h))
    draw.text(
        (panel_w * 2 + 12, header_h + panel_h - 24),
        f"t={timestamp_s:.2f}s · source={capture_fps:.1f}fps · playback={playback_speed:.1f}x",
        fill=(245, 245, 245),
        font=small,
    )
    return np.asarray(canvas)


def make_contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    selected = [paths[0], paths[len(paths) // 2], paths[-1]] if paths else []
    cell_w, cell_h, header = 520, 350, 70
    sheet = Image.new("RGB", (cell_w * 3, header + cell_h), (241, 244, 246))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 18), title, fill=(18, 27, 34), font=load_font(24, bold=True))
    for index, path in enumerate(selected):
        image = Image.open(path).convert("RGB")
        image.thumbnail((500, 330), Image.Resampling.LANCZOS)
        sheet.paste(image, (index * cell_w + 10, header + 8))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def configure_environment(
    cfg,
    wrapper_path: Path,
    anchor: dict[str, Any],
    height_scan_proxy: dict[str, Any] | None = None,
    num_envs: int = 1,
) -> None:
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 2.5
    cfg.scene.terrain.terrain_type = "usd"
    cfg.scene.terrain.usd_path = str(wrapper_path)
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.max_init_terrain_level = None
    if cfg.scene.height_scanner is not None and height_scan_proxy is not None:
        cfg.scene.height_scanner.mesh_prim_paths = [str(height_scan_proxy["mesh_prim_path"])]
    # Rough locomotion tasks normally advance a procedurally generated terrain
    # curriculum after reset.  UrbanVerse supplies a fixed USD terrain instead,
    # so retaining that training-only term dereferences a removed generator.
    cfg.curriculum.terrain_levels = None
    cfg.scene.robot.init_state.pos = tuple(anchor["spawn_position"])
    yaw = float(anchor["spawn_yaw_rad"])
    cfg.scene.robot.init_state.rot = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    cfg.scene.sky_light = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    # Velocity-command arrows are synthetic debug geometry and must not enter
    # RGB/depth perception data.
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = None
    cfg.events.add_base_mass = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.terminations.base_contact = None
    cfg.episode_length_s = 1000.0


def configure_scene10_continuous_traffic_sources(
    cfg: Any,
    registry_path: Path,
) -> list[dict[str, str]]:
    """Register source vehicles before Isaac Lab initializes Fabric/cameras.

    The traffic manager still renders independent, recentered payload proxies.
    Registering the source prims here makes their composed geometry available to
    Isaac Lab before ``gym.make`` and preserves the integration order validated
    by the Scene 10 policy-isolation run.
    """
    from isaaclab.assets import AssetBaseCfg

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = {row["asset_id"]: row for row in registry["records"]}
    configured: list[dict[str, str]] = []
    for index, asset_id in enumerate(INTEGRATED_MIXED_VEHICLE_IDS):
        if asset_id not in records:
            raise KeyError(f"traffic asset is absent from registry: {asset_id}")
        source_path = str(records[asset_id]["scene_prim_path"])
        if not source_path.startswith("/World/"):
            raise ValueError(f"unexpected Scene 10 traffic prim path: {source_path}")
        mounted_source_path = "/World/ground/terrain" + source_path[len("/World") :]
        setattr(
            cfg.scene,
            f"dynamic_traffic_vehicle_{index:02d}",
            AssetBaseCfg(
                prim_path=mounted_source_path,
                spawn=None,
                collision_group=-1,
            ),
        )
        configured.append(
            {
                "asset_id": asset_id,
                "source": mounted_source_path,
                "creation_phase": "existing source prim registered before gym.make",
            }
        )
    return configured


def main() -> int:
    args = parse_args()
    if args.derived_visualization_stride < 1 or args.full_metrics_stride < 1:
        raise ValueError("derived visualization and full metrics strides must be positive")
    if args.coda_pinhole_rig and args.strict_document_calibration:
        raise ValueError("--coda-pinhole-rig uses newly generated calibration and cannot use document calibration")
    if args.allow_scaled_document_calibration and not args.strict_document_calibration:
        raise ValueError("--allow-scaled-document-calibration requires --strict-document-calibration")
    if args.coda_delivery:
        if not (args.strict_document_calibration or args.coda_pinhole_rig):
            raise ValueError("--coda-delivery requires document calibration or --coda-pinhole-rig")
        if args.lightweight_three_panel:
            raise ValueError("--coda-delivery requires all three onboard RGB cameras")
        if not args.coda_sequence_name.startswith("seq_"):
            raise ValueError("--coda-sequence-name must use the seq_* naming convention")
        if args.capture_fps <= 0.0:
            raise ValueError("--coda-delivery requires a positive capture rate")
    camera_calibrations = (
        coda_pinhole_calibrations(
            args.sensor_width,
            args.sensor_height,
            args.coda_pinhole_horizontal_fov_deg,
            args.coda_pinhole_side_yaw_deg,
        )
        if args.coda_pinhole_rig
        else CAMERA_CALIBRATIONS
    )
    if abs(float(args.trajectory_carrier_height_offset_m)) > 1.0e-9 and args.execution_mode != "trajectory_carrier":
        raise ValueError("--trajectory-carrier-height-offset-m requires --execution-mode trajectory_carrier")
    if args.safe_shuttle and (args.route_duration_seconds is None or args.route_duration_seconds < 20.0):
        raise ValueError("--safe-shuttle requires --route-duration-seconds >= 20")
    if args.right_angle_route and args.safe_shuttle:
        raise ValueError("--right-angle-route and --safe-shuttle are mutually exclusive")
    if args.right_angle_route and args.execution_mode != "policy":
        raise ValueError("--right-angle-route requires physical policy execution")
    if args.right_angle_turn_deg is not None and not args.right_angle_route:
        raise ValueError("--right-angle-turn-deg requires --right-angle-route")
    if args.right_angle_second_leg_extent <= 0.5:
        raise ValueError("--right-angle-second-leg-extent must be greater than 0.5")
    if (args.spawn_world_xy is None) != (args.spawn_world_yaw_deg is None):
        raise ValueError("--spawn-world-xy and --spawn-world-yaw-deg must be supplied together")
    if args.spawn_world_xy is not None and abs(float(args.spawn_forward_offset)) > 1e-9:
        raise ValueError("explicit world spawn and forward spawn offset are mutually exclusive")
    if args.phase1_dynamic_vehicle:
        if args.execution_mode != "policy":
            raise ValueError("--phase1-dynamic-vehicle requires physical policy execution")
        if args.scene != "scene_10_cbd_cross_intersection_diverse_obstacles":
            raise ValueError("phase one is currently qualified only for scene_10_cbd")
        if args.phase1_vehicle_distance <= 0.0 or args.phase1_vehicle_speed <= 0.0:
            raise ValueError("phase-one vehicle distance and speed must be positive")
    portable_scene = None
    if args.scene10_continuous_traffic:
        if args.phase1_dynamic_vehicle:
            raise ValueError("continuous Scene 10 traffic and phase-one single vehicle are mutually exclusive")
        if args.execution_mode not in {"policy", "trajectory_carrier"}:
            raise ValueError(
                "continuous traffic requires policy or prescribed trajectory-carrier execution"
            )
        portable_scene = (
            TrafficSceneConfig.load(args.traffic_scene_config)
            if args.traffic_scene_config is not None
            else None
        )
        if portable_scene is not None:
            if portable_scene.scene_id != args.scene:
                raise ValueError(
                    f"traffic scene config {portable_scene.scene_id!r} does not match {args.scene!r}"
                )
            args.traffic_registry = portable_scene.vehicle_catalog
            args.traffic_automotive_routes = portable_scene.automotive_routes
        elif args.scene != "scene_10_cbd_cross_intersection_diverse_obstacles":
            raise ValueError("non-Scene10 continuous traffic requires --traffic-scene-config")
        required_traffic_paths = {
            "traffic_registry": args.traffic_registry,
            "traffic_automotive_routes": args.traffic_automotive_routes,
        }
        if portable_scene is None:
            required_traffic_paths.update(
                traffic_audit_inventory=args.traffic_audit_inventory,
                traffic_validated_routes=args.traffic_validated_routes,
            )
        missing = [name for name, path in required_traffic_paths.items() if path is None]
        if missing:
            raise ValueError(f"continuous traffic requires paths: {', '.join(missing)}")
        absent = [name for name, path in required_traffic_paths.items() if not path.resolve().is_file()]
        if absent:
            raise FileNotFoundError(f"continuous traffic inputs are absent: {', '.join(absent)}")
        if args.minimum_go2_vehicle_clearance < 0.0:
            raise ValueError("--minimum-go2-vehicle-clearance must be non-negative")
        if args.maximum_traffic_visual_center_error <= 0.0:
            raise ValueError("--maximum-traffic-visual-center-error must be positive")
        if args.maximum_rendered_body_axis_error <= 0.0:
            raise ValueError("--maximum-rendered-body-axis-error must be positive")
        if args.minimum_visible_vehicle_pass_duration < 0.0:
            raise ValueError("--minimum-visible-vehicle-pass-duration must be non-negative")
        minimum_vehicle_count = 1 if portable_scene is not None else 5
        if args.traffic_vehicle_count < minimum_vehicle_count:
            raise ValueError(
                f"continuous traffic requires at least {minimum_vehicle_count} vehicles"
            )
    started = time.perf_counter()
    np.random.seed(args.seed)
    run_dir = args.run_dir.resolve()
    captures = run_dir / "captures"
    metadata = run_dir / "metadata"
    visualizations = run_dir / "visualizations"
    videos = run_dir / "videos"
    wrappers = run_dir / "wrapper"
    for directory in (captures, metadata, visualizations, videos, wrappers):
        directory.mkdir(parents=True, exist_ok=True)
    summary_path = metadata / "summary.json"
    source_usd = args.usd.resolve()
    source_tar = args.tar.resolve() if args.tar else None
    policy_path = args.policy.resolve()
    checkpoint_path = args.checkpoint.resolve()
    route_config = json.loads(args.route_config.read_text(encoding="utf-8"))
    navigation_config = navigation.resolve_scene_navigation(route_config, args.scene)
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive")
    if args.overview_chase_distance <= 0.0 or args.overview_chase_height <= 0.0:
        raise ValueError("overview chase distance and height must be positive")
    if args.overview_focal_length <= 0.0:
        raise ValueError("--overview-focal-length must be positive")
    if args.external_route_lookahead_distance <= 0.0:
        raise ValueError("--external-route-lookahead-distance must be positive")
    if args.external_route_minimum_tracking_speed < 0.0:
        raise ValueError("--external-route-minimum-tracking-speed must be non-negative")
    external_route: dict[str, Any] | None = None
    external_waypoints: list[list[float]] | None = None
    if args.reference_route_file is not None:
        route_path = args.reference_route_file.resolve()
        external_route = json.loads(route_path.read_text(encoding="utf-8"))
        if external_route.get("scene") != args.scene:
            raise ValueError(
                f"reference route scene {external_route.get('scene')!r} does not match {args.scene!r}"
            )
        external_waypoints = [
            [float(point[0]), float(point[1])] for point in external_route.get("points_xy", [])
        ]
        if len(external_waypoints) < 2:
            raise ValueError("reference route needs at least two points_xy")
        navigation_config.update(
            {
                "mode": "waypoint_loop",
                "tracking_controller": "pure_pursuit",
                "stop_at_final_waypoint": True,
                "stuck_window_s": 10.0,
                "stuck_command_yaw_rate": 0.10,
                "stuck_confirmation_s": 20.0,
            }
        )
        pure_pursuit = navigation_config.setdefault("pure_pursuit", {})
        pure_pursuit.update(
            {
                "max_forward_speed": 0.34,
                "max_tracking_yaw_rate": 0.18,
                "minimum_tracking_speed": args.external_route_minimum_tracking_speed,
                "lookahead_distance_m": args.external_route_lookahead_distance,
                "curvature_speed_gain": 2.0,
                "vx_rate_limit": 1.2,
                "yaw_rate_limit": 0.8,
                "goal_tolerance": 0.35,
                "deceleration_distance": 0.80,
            }
        )
        navigation_config["runtime_external_reference_route"] = {
            "source": str(route_path),
            "source_sha256": sha256(route_path),
            "point_count": len(external_waypoints),
            "route_length_stage_units": external_route.get("route_length_stage_units"),
            "static_physx_prevalidated": bool(
                (external_route.get("planner") or {}).get("inflated_grid_all_points_valid")
            ),
            "controller": "pure_pursuit",
            "lookahead_distance_m": args.external_route_lookahead_distance,
            "minimum_tracking_speed_mps": args.external_route_minimum_tracking_speed,
            "stuck_window_s": 10.0,
            "stuck_command_yaw_rate": 0.10,
            "stuck_confirmation_s": 20.0,
        }
    if args.road_route_extent is not None:
        if args.road_route_extent <= 0.5:
            raise ValueError("--road-route-extent must be greater than 0.5 stage units")
        road_probe_config = navigation_config.get("road_corridor_probe")
        if not isinstance(road_probe_config, dict) or not road_probe_config.get("enabled"):
            raise ValueError("--road-route-extent requires an enabled road_corridor_probe")
        original_extent = float(road_probe_config.get("route_extent_stage_units", 2.5))
        road_probe_config["route_extent_stage_units"] = float(args.road_route_extent)
        navigation_config["runtime_road_route_extent_override"] = {
            "original_stage_units": original_extent,
            "effective_stage_units": float(args.road_route_extent),
            "reason": "explicit physical-locomotion route admission override",
        }
    global_route_config = navigation_config.get("global_route_planner")
    if args.global_route:
        if not isinstance(global_route_config, dict) or not global_route_config:
            raise ValueError("--global-route requires a global_route_planner block in the navigation config")
        navigation_config["mode"] = "global_planned"
    if str(navigation_config.get("mode", "waypoint_loop")) == "global_planned":
        if not isinstance(global_route_config, dict) or not global_route_config:
            raise ValueError("mode global_planned requires a global_route_planner block in the navigation config")
        if args.global_route_extent is not None:
            if args.global_route_extent <= 1.0:
                raise ValueError("--global-route-extent must be greater than 1.0 metres")
            global_route_config["grid_half_extent_m"] = float(args.global_route_extent)
        if args.global_route_target_length is not None:
            target_band = [float(args.global_route_target_length[0]), float(args.global_route_target_length[1])]
            if target_band[0] <= 0.0 or target_band[1] < target_band[0]:
                raise ValueError("--global-route-target-length MIN_M MAX_M must be increasing positive metres")
            global_route_config["target_route_length_m"] = target_band
        else:
            calibrated_band = calibrate_global_route_length(
                args.scene,
                global_route_config,
                args.route_duration_seconds,
                float(navigation_config.get("max_forward_speed", 0.32)),
            )
            if calibrated_band is not None:
                global_route_config["target_route_length_m"] = calibrated_band
        if args.global_route_seed is not None:
            global_route_config["sample_seed"] = int(args.global_route_seed)
        if args.global_route_min_turn is not None:
            global_route_config["min_total_turn_rad"] = float(args.global_route_min_turn)
        if portable_scene is not None:
            joint_constraints = build_joint_route_constraints(
                automotive_routes_path=portable_scene.automotive_routes,
                vehicle_catalog_path=portable_scene.vehicle_catalog,
                pedestrian_config_path=portable_scene.pedestrian_config,
                go2_radius_m=float(global_route_config.get("robot_radius_m", 0.40)),
                vehicle_clearance_m=float(
                    global_route_config.get(
                        "vehicle_route_clearance_m",
                        max(0.50, float(args.minimum_go2_vehicle_clearance)),
                    )
                ),
                pedestrian_radius_m=float(
                    global_route_config.get("pedestrian_radius_m", 0.35)
                ),
                pedestrian_clearance_m=float(
                    global_route_config.get("pedestrian_route_clearance_m", 0.25)
                ),
            )
            global_route_config["joint_route_constraints"] = joint_constraints
            write_json(metadata / "go2_joint_route_constraints.json", joint_constraints)
            navigation_config["runtime_joint_route_constraints"] = {
                "source": "exact automotive_routes and pedestrian_config from TrafficSceneConfig",
                "policy": "hard_exclusion",
                "vehicle_constraint_count": joint_constraints["counts"]["vehicle"],
                "pedestrian_constraint_count": joint_constraints["counts"]["pedestrian"],
                "evidence": str(metadata / "go2_joint_route_constraints.json"),
            }
        navigation_config["global_route_planner"] = global_route_config
    if args.policy_kind in EXTERNAL_KINDS and args.locomotion_profile != "flat":
        raise ValueError("External Go2 policies require --locomotion-profile flat")
    controller_profile = navigation.CONTROLLER_PROFILES[args.locomotion_profile]
    schedule = schedule_for_scene(
        route_config,
        args.turn_sign,
        args.route_duration_scale,
        args.route_duration_seconds,
    )
    height_scan_proxy = navigation_config.get("rough_height_scan_proxy")
    if args.rough_height_scan_ground_z_stage_units is not None:
        height_scan_proxy = {
            "enabled": True,
            "ground_z_stage_units": float(args.rough_height_scan_ground_z_stage_units),
            "extent_stage_units": float(args.rough_height_scan_extent_stage_units),
            "mesh_prim_path": "/World/ground/terrain/HeightScanProxy",
            "source": "Explicit per-run CLI override from the validated reference-route ground_z.",
            "limitation": (
                "Invisible non-colliding flat height observation for the frozen rough locomotion policy only; "
                "RGB, depth, contacts and route admission still use the source scene."
            ),
        }
    if args.locomotion_profile != "rough" or not bool((height_scan_proxy or {}).get("enabled", False)):
        height_scan_proxy = None
    wrapper_path = wrappers / f"{args.scene}_go2_route_wrapper.usda"
    if height_scan_proxy is None:
        wrapper_text = args.wrapper_template.read_text(encoding="utf-8")
    else:
        proxy_template = PROJECT_ROOT / "configs" / "wrappers" / "urbanverse_reference_with_height_scan_proxy.usda.in"
        wrapper_text = proxy_template.read_text(encoding="utf-8")
        wrapper_text = wrapper_text.replace(
            "HEIGHT_SCAN_PROXY_EXTENT", str(float(height_scan_proxy["extent_stage_units"]))
        ).replace("HEIGHT_SCAN_PROXY_Z", str(float(height_scan_proxy["ground_z_stage_units"])))
    wrapper_path.write_text(wrapper_text.replace("SOURCE_USD", str(source_usd)), encoding="utf-8")
    summary: dict[str, Any] = {
        "status": "failed",
        "scene": args.scene,
        "family": args.family,
        "replicate": args.replicate,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "Go2 high-level route execution and model-agnostic raw-run capture; no locomotion training",
    }
    write_json(summary_path, summary)
    simulation_app = None
    env = None
    camera_rows: list[dict[str, Any]] = []
    strict_maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    strict_mapping_metadata: dict[str, Any] = {}
    frame_paths: list[Path] = []
    preauthored_traffic_sources: list[dict[str, str]] = []
    pedestrian_config: dict[str, Any] | None = None
    preauthored_people: list[dict[str, str]] = []
    go2_runtime_support: dict[str, Any] | None = None
    traffic_manager: Scene10ContinuousVehicleManager | None = None
    people_manager: OfficialPeopleManager | None = None
    coda_writer: CodaDatasetWriter | None = None
    coda_lidar = None
    coda_lidar_annotator = None
    coda_lidar_accumulator: RtxLidarAccumulator | None = None
    coda_camera_names = {
        "center_forward_pinhole": "cam_front",
        "left_fisheye": "cam_front_left",
        "right_fisheye": "cam_front_right",
        "left_forward_pinhole": "cam_front_left",
        "right_forward_pinhole": "cam_front_right",
    }
    coda_lidar_scan_count = 0
    coda_camera_lidar_offsets_ns: list[int] = []
    coda_validation: dict[str, Any] | None = None
    result_code = 1
    try:
        if not source_usd.is_file() or not policy_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                {"source_usd": str(source_usd), "policy": str(policy_path), "checkpoint": str(checkpoint_path)}
            )
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(
            headless=True,
            enable_cameras=True,
            device=f"cuda:{args.gpu}",
            renderer="RayTracedLighting",
            experience=str(PROJECT_ROOT / "configs" / "isaac45_urbanverse_go2_headless.kit"),
            width=args.overview_width,
            height=args.overview_height,
            multi_gpu=False,
        )
        simulation_app = launcher.app

        import carb
        import cv2  # noqa: F401
        import gymnasium as gym
        import omni.kit.app
        import omni.physx
        import omni.usd
        import torch
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Usd, UsdGeom
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        settings = carb.settings.get_settings()
        settings.set("/rtx/multiGpu/enabled", False)
        if args.phase1_dynamic_vehicle or args.scene10_continuous_traffic:
            # Runtime vehicle transforms do not provide stable temporal motion
            # history in this Isaac 4.5 import.  FXAA avoids DLSS/TAA ghosting,
            # while explicit motion-blur disable keeps the original mesh sharp.
            settings.set("/rtx/post/aa/op", 2)
            settings.set("/rtx/post/motionblur/enabled", bool(args.dynamic_agent_motion_blur))
            settings.set("/rtx/post/motionBlur/enabled", bool(args.dynamic_agent_motion_blur))
        settings.set("/physics/cudaDevice", args.gpu)
        settings.set("/renderer/activeGpu", args.gpu)
        enable_extension("omni.kit.asset_converter")
        if args.coda_delivery:
            enable_extension("isaacsim.sensors.rtx")
        for _ in range(3):
            simulation_app.update()

        source_matrix, source_camera_meta = source_camera_matrix(source_usd, args.camera)
        anchor = derive_route_anchor(source_matrix, route_config["route_anchor_policy"])
        heading_offset_rad = float(navigation_config.get("heading_offset_rad", 0.0))
        anchor["spawn_yaw_rad"] = navigation.wrap_angle(float(anchor["spawn_yaw_rad"]) + heading_offset_rad)
        anchor["navigation_heading_offset_rad"] = heading_offset_rad
        anchor["derivation"] += "; scene navigation heading offset applied before Go2 spawn"
        if external_waypoints is not None:
            first = np.asarray(external_waypoints[0], dtype=np.float64)
            second = np.asarray(external_waypoints[1], dtype=np.float64)
            route_yaw = math.atan2(float(second[1] - first[1]), float(second[0] - first[0]))
            anchor["spawn_position"][0] = float(first[0])
            anchor["spawn_position"][1] = float(first[1])
            anchor["spawn_yaw_rad"] = route_yaw
            anchor["derivation"] += "; start XY/yaw loaded from the validated external reference route"
        global_route_spawn_override = None
        if str(navigation_config.get("mode", "waypoint_loop")) == "global_planned":
            planner_cfg = navigation_config.get("global_route_planner") or {}
            override_xy = planner_cfg.get("start_position_override_m")
            if isinstance(override_xy, list) and len(override_xy) == 2:
                global_route_spawn_override = {
                    "original_spawn_position": [float(value) for value in anchor["spawn_position"]],
                    "original_heading_world_rad": float(anchor["spawn_yaw_rad"]),
                    "override_xy_m": [float(override_xy[0]), float(override_xy[1])],
                    "override_yaw_rad": planner_cfg.get("start_yaw_override_rad"),
                }
                anchor["spawn_position"][0] = float(override_xy[0])
                anchor["spawn_position"][1] = float(override_xy[1])
                if planner_cfg.get("start_yaw_override_rad") is not None:
                    anchor["spawn_yaw_rad"] = navigation.wrap_angle(float(planner_cfg["start_yaw_override_rad"]))
                navigation_config["runtime_global_route_spawn_override"] = {
                    "original_spawn_position": global_route_spawn_override["original_spawn_position"],
                    "original_heading_world_rad": global_route_spawn_override["original_heading_world_rad"],
                    "effective_spawn_position_before_physx_ground_correction": [
                        float(value) for value in anchor["spawn_position"]
                    ],
                    "effective_heading_world_rad": float(anchor["spawn_yaw_rad"]),
                    "reason": (
                        "The recorded spawn sits inside the planned 0.55 m clearance of a market-stall "
                        "obstacle; the global-planned start is advanced east onto the clear road band "
                        "(esdf ~2 m) so the route's first leg stays in the spawn-heading direction and "
                        "no large in-place turn is required."
                    ),
                }
                anchor["derivation"] += "; global_planned start_position_override_m applied"
        if args.spawn_world_xy is not None:
            original_spawn = [float(value) for value in anchor["spawn_position"]]
            original_yaw = float(anchor["spawn_yaw_rad"])
            anchor["spawn_position"][0] = float(args.spawn_world_xy[0])
            anchor["spawn_position"][1] = float(args.spawn_world_xy[1])
            anchor["spawn_yaw_rad"] = navigation.wrap_angle(math.radians(float(args.spawn_world_yaw_deg)))
            navigation_config["runtime_explicit_world_spawn"] = {
                "original_spawn_position": original_spawn,
                "original_heading_world_rad": original_yaw,
                "effective_spawn_position_before_physx_ground_correction": [
                    float(value) for value in anchor["spawn_position"]
                ],
                "effective_heading_world_rad": float(anchor["spawn_yaw_rad"]),
                "reason": "place the physical preview on a selected source-scene road branch",
                "physx_route_admission_required": True,
                "source_usd_modified": False,
            }
            anchor["derivation"] += "; explicit recorded world road spawn applied before PhysX admission"
        elif abs(float(args.spawn_forward_offset)) > 1e-9:
            original_spawn = [float(value) for value in anchor["spawn_position"]]
            spawn_yaw = float(anchor["spawn_yaw_rad"])
            anchor["spawn_position"][0] = original_spawn[0] + math.cos(spawn_yaw) * float(
                args.spawn_forward_offset
            )
            anchor["spawn_position"][1] = original_spawn[1] + math.sin(spawn_yaw) * float(
                args.spawn_forward_offset
            )
            navigation_config["runtime_spawn_forward_offset"] = {
                "stage_units": float(args.spawn_forward_offset),
                "original_spawn_position": original_spawn,
                "effective_spawn_position_before_physx_ground_correction": [
                    float(value) for value in anchor["spawn_position"]
                ],
                "heading_world_rad": spawn_yaw,
                "reason": "place the physical preview on the same admitted road nearer the target junction",
                "source_usd_modified": False,
            }
            anchor["derivation"] += "; recorded runtime forward offset applied along the road heading"
        light_path_scales: dict[str, float] = {}
        for item in args.collection_light_path_scale:
            if "=" not in item:
                raise ValueError("--collection-light-path-scale must use PRIM_PATH=SCALE")
            prim_path, raw_scale = item.rsplit("=", 1)
            if prim_path in light_path_scales:
                raise ValueError(f"duplicate collection light path scale: {prim_path}")
            light_path_scales[prim_path] = float(raw_scale)
        light_scale_overrides = author_all_light_scale(
            wrapper_path,
            args.collection_light_scale,
            dome_scale=args.collection_dome_light_scale,
            distant_scale=args.collection_distant_light_scale,
            sphere_scale=args.collection_sphere_light_scale,
            strong_sphere_scale=args.collection_strong_sphere_light_scale,
            strong_sphere_intensity_threshold=args.collection_strong_sphere_intensity_threshold,
            path_scales=light_path_scales,
            source_dome_background_visible=args.source_dome_background_visible,
            source_dome_texture_enabled=args.source_dome_texture_enabled,
        )
        write_json(metadata / "collection_light_scale.json", light_scale_overrides)
        safe_print(
            "COLLECTION_LIGHT_SCALE "
            + json.dumps(
                {
                    "scale": light_scale_overrides["scale"],
                    "requested_scales": light_scale_overrides["requested_scales"],
                    "path_scales": light_scale_overrides["path_scales"],
                    "adjusted_count": light_scale_overrides["adjusted_count"],
                    "type_counts": light_scale_overrides["type_counts"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        source_stage = Usd.Stage.Open(str(source_usd), load=Usd.Stage.LoadNone)
        source_stage_metadata = {
            "up_axis": str(UsdGeom.GetStageUpAxis(source_stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(source_stage)),
            "default_prim": str(source_stage.GetDefaultPrim().GetPath()) if source_stage.GetDefaultPrim() else None,
        }
        official_task = str(controller_profile["task"])
        cfg = parse_env_cfg(official_task, device=f"cuda:{args.gpu}", num_envs=1)
        if args.scene10_continuous_traffic and args.traffic_scene_config is not None:
            preauthored_traffic_sources = author_portable_traffic_visuals(
                wrapper_path,
                args.traffic_registry.resolve(),
                args.traffic_vehicle_count,
            )
        if portable_scene is not None and portable_scene.pedestrian_config is not None:
            pedestrian_config, preauthored_people = author_official_people(
                wrapper_path,
                portable_scene.pedestrian_config,
            )
        configure_environment(cfg, wrapper_path, anchor, height_scan_proxy)
        if args.policy_kind in EXTERNAL_KINDS:
            from urbanverse.dynamic_agents.navigation.external_go2_policy import configure_external_go2_environment
            configure_external_go2_environment(cfg, args.external_policy_source, args.policy_kind)
        if args.scene10_continuous_traffic:
            if preauthored_traffic_sources:
                configure_preauthored_traffic_visuals(cfg, preauthored_traffic_sources)
            else:
                preauthored_traffic_sources = configure_scene10_continuous_traffic_sources(
                    cfg,
                    args.traffic_registry.resolve(),
                )
        if preauthored_people:
            configure_preauthored_people(cfg, preauthored_people)
        if external_route is not None:
            go2_runtime_support = configure_go2_support_corridor(cfg, external_route)
            if go2_runtime_support is not None:
                write_json(metadata / "go2_runtime_physics_support.json", go2_runtime_support)
        pre_camera_application = route_camera_application(args.appearance_profile, args.seed)
        camera_definitions = add_navigation_camera_sensors(
            cfg, args, pre_camera_application, camera_calibrations
        )
        cfg.seed = args.seed
        cfg.sim.enable_scene_query_support = True
        cfg.sim.render_interval = cfg.decimation
        env = gym.make(official_task, cfg=cfg)
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene["contact_forces"]
        dt = float(unwrapped.step_dt)
        scene_query = omni.physx.get_physx_scene_query_interface()
        ground_probe = probe_route_ground(anchor, scene_query)
        selected_ground_z = ground_probe["selected_ground_z"]
        spawn_correction = {
            "applied": selected_ground_z is not None,
            "original_spawn_position": list(anchor["spawn_position"]),
            "reason": "PhysX height-cluster route support replaces the camera-height estimate; ambiguous multi-level hits may also recenter XY on the selected support cluster",
        }
        if selected_ground_z is not None:
            selected_xy = ground_probe.get("selected_ground_xy")
            recenter_xy = bool(selected_xy is not None and float(ground_probe.get("all_usable_z_span") or 0.0) > 0.50)
            corrected_spawn = [
                float(selected_xy[0]) if recenter_xy else float(anchor["spawn_position"][0]),
                float(selected_xy[1]) if recenter_xy else float(anchor["spawn_position"][1]),
                float(selected_ground_z) + float(route_config["route_anchor_policy"]["go2_initial_base_height_stage_units"]),
            ]
            yaw = float(anchor["spawn_yaw_rad"])
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] = torch.tensor(corrected_spawn, dtype=root_state.dtype, device=root_state.device)
            root_state[:, 3:7] = torch.tensor(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                dtype=root_state.dtype,
                device=root_state.device,
            )
            root_state[:, 7:] = 0.0
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_position = robot.data.default_joint_pos.clone()
            joint_velocity = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_position, joint_velocity)
            robot.set_joint_position_target(joint_position)
            robot.set_joint_velocity_target(joint_velocity)
            robot.reset()
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
            unwrapped.scene.update(dt)
            observations = unwrapped.observation_manager.compute(update_history=True)
            anchor["spawn_position"] = corrected_spawn
            anchor["estimated_ground_z"] = float(selected_ground_z)
            anchor["derivation"] += "; support corrected from clustered nearby PhysX ray probes"
            spawn_correction["corrected_spawn_position"] = corrected_spawn
            spawn_correction["xy_recentered"] = recenter_xy
        else:
            spawn_correction["corrected_spawn_position"] = list(anchor["spawn_position"])
        ground_probe["spawn_correction"] = spawn_correction
        total_duration = (
            float(args.route_duration_seconds)
            if args.route_duration_seconds is not None
            else sum(float(item["duration_s"]) for item in schedule)
        )
        total_steps = int(round(total_duration / dt))
        capture_stride = max(1, int(round(1.0 / (args.capture_fps * dt))))
        actual_capture_fps = 1.0 / (capture_stride * dt)
        command_term = unwrapped.command_manager.get_term("base_velocity")
        external_policy_metadata = None
        if args.policy_kind in EXTERNAL_KINDS:
            from urbanverse.dynamic_agents.navigation.external_go2_policy import ExternalGo2Policy
            external = ExternalGo2Policy(args.external_policy_source, args.policy_kind, robot.joint_names,
                                         unwrapped.device, model_path=policy_path)
            external_policy_metadata = {"upstream_config": external.cfg, "policy_joint_names": external.policy_names,
                "sim_joint_names": robot.joint_names, "policy_to_sim_indices": external.indices,
                "source_manifest": json.loads((args.external_policy_source / "manifest.json").read_text())}
            def policy(observation):
                return external.act(robot, command_term.vel_command_b)[0]
        else:
            policy = torch.jit.load(str(policy_path), map_location=f"cuda:{args.gpu}").eval()
        command_term.time_left[:] = 1000.0
        command_term.is_heading_env[:] = False
        command_term.is_standing_env[:] = False
        controller_warmup_duration_s = (
            0.0
            if args.execution_mode == "trajectory_carrier"
            else float(navigation_config.get("controller_warmup_duration_s", 1.0))
        )
        controller_warmup_steps = max(0, int(round(controller_warmup_duration_s / dt)))
        controller_warmup_path = metadata / "controller_warmup.jsonl"
        warmup_collision_steps = 0
        warmup_start_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        with controller_warmup_path.open("w", encoding="utf-8") as warmup_handle:
            for warmup_index in range(controller_warmup_steps):
                command_term.vel_command_b.zero_()
                command_term.time_left[:] = 1000.0
                command_term.is_heading_env[:] = False
                command_term.is_standing_env[:] = False
                observation_tensor = observations["policy"] if isinstance(observations, dict) else observations
                with torch.inference_mode():
                    warmup_actions = policy(observation_tensor)
                    observations, _, warmup_terminated, warmup_truncated, _ = env.step(warmup_actions)
                warmup_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
                warmup_quaternion = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
                warmup_contact = contact_sensor.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
                warmup_contact_norms = np.linalg.norm(warmup_contact, axis=-1)
                warmup_classification = navigation.classify_contacts(
                    contact_sensor.body_names, warmup_contact_norms
                )
                warmup_collision_steps += int(bool(warmup_classification["non_foot_collision"]))
                append_jsonl(
                    warmup_handle,
                    {
                        "warmup_step_index": warmup_index + 1,
                        "timestamp_s": (warmup_index + 1) * dt,
                        "control_command_body": [0.0, 0.0, 0.0],
                        "action_joint_position_policy": warmup_actions[0].detach().cpu().numpy().tolist(),
                        "base_position_world": warmup_position.tolist(),
                        "base_quaternion_wxyz_world": warmup_quaternion.tolist(),
                        "contact_classification": warmup_classification,
                        "terminated": bool(warmup_terminated[0].item()),
                        "truncated": bool(warmup_truncated[0].item()),
                    },
                )
        warmup_final_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        controller_warmup = {
            "duration_s": controller_warmup_steps * dt,
            "step_count": controller_warmup_steps,
            "command": [0.0, 0.0, 0.0],
            "purpose": (
                "Skipped because the trajectory carrier prescribes the base pose from the first raw timestamp."
                if args.execution_mode == "trajectory_carrier"
                else "Let the frozen locomotion policy establish a supported stance before raw navigation timestamps begin."
            ),
            "trace": str(controller_warmup_path),
            "start_base_position": warmup_start_position.tolist(),
            "final_base_position": warmup_final_position.tolist(),
            "non_foot_collision_step_count": warmup_collision_steps,
            "excluded_from_navigation_run": True,
        }
        initial_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        initial_quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        initial_yaw = yaw_from_quat(initial_quat)
        route_mode = str(navigation_config.get("mode", "waypoint_loop"))
        if route_mode not in ("waypoint_loop", "global_planned"):
            raise ValueError(f"unsupported navigation mode: {route_mode}")
        road_probe_config = navigation_config.get("road_corridor_probe")
        road_route_probe = None
        global_route_plan = None
        safe_shuttle_plan = None
        right_angle_route_probe = None
        if external_waypoints is not None:
            waypoints_world = external_waypoints
            navigation_config["stop_at_final_waypoint"] = True
            write_json(
                metadata / "external_reference_route.json",
                {
                    "accepted": bool(
                        external_route
                        and (external_route.get("planner") or {}).get("inflated_grid_all_points_valid")
                    ),
                    "source": navigation_config["runtime_external_reference_route"],
                    "points_xy": external_waypoints,
                },
            )
        elif route_mode == "global_planned":
            if selected_ground_z is None:
                raise RuntimeError("global route planning requires a PhysX ground height")
            global_route_cache_dir = (
                args.global_route_cache_dir.resolve()
                if args.global_route_cache_dir is not None
                else PROJECT_ROOT / "cache" / "global_route"
            )

            def global_raycast(origin_xyz, direction_xyz, max_distance) -> dict[str, Any]:
                result = scene_query.raycast_closest(
                    carb.Float3(*origin_xyz), carb.Float3(*direction_xyz), float(max_distance)
                )
                row: dict[str, Any] = {"hit": False}
                if isinstance(result, dict) and bool(result.get("hit", False)):
                    position = result.get("position")
                    collision = str(result.get("collision") or "")
                    rigid_body = str(result.get("rigidBody") or "")
                    row.update(
                        {
                            "hit": True,
                            "position": [float(position[index]) for index in range(3)] if position is not None else None,
                            "distance": float(result.get("distance")) if result.get("distance") is not None else None,
                            "collision": collision,
                            "rigid_body": rigid_body,
                            "robot_hit": "/Robot" in collision or "/Robot" in rigid_body,
                        }
                    )
                return row

            global_route_plan = global_planner.plan_global_route(
                [float(initial_position[0]), float(initial_position[1])],
                float(selected_ground_z),
                global_raycast,
                omni.usd.get_context().get_stage(),
                navigation_config.get("global_route_planner", {}),
                cache_dir=str(global_route_cache_dir),
                cache_key=str(source_usd),
                spawn_heading_rad=float(initial_yaw),
            )
            write_json(metadata / "global_route_plan.json", global_route_plan)
            safe_print(
                "GLOBAL_ROUTE_PLAN "
                + json.dumps(
                    {
                        "accepted": global_route_plan["accepted"],
                        "reason": global_route_plan.get("reason"),
                        "path_length_m": (global_route_plan.get("astar") or {}).get("path_length_m"),
                        "target_route_length_m": global_route_plan.get("target_route_length_m"),
                        "length_band_used_m": global_route_plan.get("length_band_used_m"),
                        "max_curvature_rad_per_m": (global_route_plan.get("spline") or {}).get(
                            "max_curvature_rad_per_m"
                        ),
                        "total_turn_rad": (global_route_plan.get("turn") or {}).get("total_turn_rad"),
                        "total_turn_deg": (global_route_plan.get("turn") or {}).get("total_turn_deg"),
                        "min_total_turn_rad": (global_route_plan.get("turn") or {}).get("min_total_turn_rad"),
                        "validation_segments": (global_route_plan.get("validation") or {}).get("segment_count"),
                        "cache_hit": (global_route_plan.get("grid") or {}).get("cache_hit"),
                        "unit_conversion": global_route_plan.get("unit_conversion"),
                        "waypoint_count": len(global_route_plan.get("waypoints_world_xy", [])),
                        "evidence": str(metadata / "global_route_plan.json"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not global_route_plan["accepted"]:
                raise RuntimeError(
                    "global route planner admitted no route: " + str(global_route_plan.get("reason"))
                )
            waypoints_world = global_route_plan["waypoints_world_xy"]
            navigation_config["stop_at_final_waypoint"] = True
            endpoint = global_route_plan["waypoints_world_xy"][-1]
            navigation_config["runtime_global_route"] = {
                "applied": True,
                "first_leg_heading_world_rad": (
                    math.atan2(
                        float(waypoints_world[1][1] - waypoints_world[0][1]),
                        float(waypoints_world[1][0] - waypoints_world[0][0]),
                    )
                    if len(waypoints_world) >= 2
                    else None
                ),
                "spawn_heading_world_rad": float(initial_yaw),
                "waypoint_count": len(waypoints_world),
                "reason": (
                    f"The route's first leg is biased toward the Go2's recorded spawn heading "
                    f"so the frozen gait only walks forward along the validated spline and "
                    f"never needs a large pure in-place turn. min_total_turn_rad "
                    f"{(global_route_plan.get('turn') or {}).get('min_total_turn_rad')} enforces "
                    f"the requested cumulative heading change on the delivered polyline."
                ),
            }
        elif road_probe_config and bool(road_probe_config.get("enabled", False)):
            if selected_ground_z is None:
                raise RuntimeError("road corridor probe requires a PhysX ground height")
            road_route_probe = probe_safe_road_corridor(
                initial_position[:2], float(selected_ground_z), initial_yaw, scene_query, road_probe_config
            )
            write_json(metadata / "road_route_probe.json", road_route_probe)
            safe_print(
                "ROAD_ROUTE_PROBE "
                + json.dumps(
                    {
                        "accepted": road_route_probe["accepted"],
                        "accepted_candidate_count": road_route_probe["accepted_candidate_count"],
                        "selected_heading_offset_deg": road_route_probe["selected_heading_offset_deg"],
                        "selected_endpoint_world_xy": road_route_probe["selected_endpoint_world_xy"],
                        "evidence": str(metadata / "road_route_probe.json"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not road_route_probe["accepted"]:
                raise RuntimeError("no flat obstacle-free road corridor passed the configured PhysX route probe")
            endpoint = road_route_probe["selected_endpoint_world_xy"]
            if args.right_angle_route:
                right_angle_route_probe = probe_right_angle_road_route(
                    initial_position[:2],
                    float(selected_ground_z),
                    initial_yaw,
                    scene_query,
                    road_probe_config,
                    road_route_probe,
                    list(args.right_angle_corner_distances),
                    float(args.right_angle_second_leg_extent),
                    args.right_angle_turn_deg,
                )
                write_json(metadata / "right_angle_route_probe.json", right_angle_route_probe)
                safe_print(
                    "RIGHT_ANGLE_ROUTE_PROBE "
                    + json.dumps(
                        {
                            "accepted": right_angle_route_probe["accepted"],
                            "accepted_candidate_count": right_angle_route_probe["accepted_candidate_count"],
                            "waypoints_world_xy": right_angle_route_probe["waypoints_world_xy"],
                            "selected_turn_angle_deg": (
                                right_angle_route_probe.get("selected") or {}
                            ).get("turn_angle_deg"),
                            "evidence": str(metadata / "right_angle_route_probe.json"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if not right_angle_route_probe["accepted"]:
                    raise RuntimeError("no two-leg right-angle road route passed the configured PhysX probes")
                nominal_right_angle_waypoints = [
                    list(point) for point in right_angle_route_probe["waypoints_world_xy"]
                ]
                arc_radius = float(navigation_config.get("right_angle_arc_radius", 1.5))
                arc_samples = int(navigation_config.get("right_angle_arc_samples", 12))
                waypoints_world = navigation.rounded_right_angle_waypoints(
                    *nominal_right_angle_waypoints,
                    radius=arc_radius,
                    sample_count=arc_samples,
                )
                navigation_config["waypoint_tolerance"] = float(
                    navigation_config.get("right_angle_waypoint_tolerance", 0.12)
                )
                navigation_config["stop_at_final_waypoint"] = True
                navigation_config["runtime_right_angle_route"] = {
                    "enabled": True,
                    "corner_distances_stage_units": list(args.right_angle_corner_distances),
                    "second_leg_extent_stage_units": float(args.right_angle_second_leg_extent),
                    "requested_turn_deg": args.right_angle_turn_deg,
                    "nominal_waypoints_world_xy": nominal_right_angle_waypoints,
                    "rounded_execution_waypoints_world_xy": waypoints_world,
                    "arc_radius_stage_units": arc_radius,
                    "arc_sample_count": arc_samples,
                }
            elif args.safe_shuttle:
                safe_shuttle_plan = build_safe_shuttle_plan(
                    initial_position[:2],
                    road_route_probe,
                    bool(navigation_config.get("safe_shuttle_secondary_spoke_enabled", False)),
                )
                waypoints_world = [initial_position[:2].tolist()] + [
                    list(leg["target_world_xy"]) for leg in safe_shuttle_plan["legs"]
                ]
                write_json(metadata / "safe_shuttle_plan.json", safe_shuttle_plan)
            elif bool(navigation_config.get("stop_at_final_waypoint", False)):
                waypoints_world = [initial_position[:2].tolist(), list(endpoint)]
            else:
                waypoints_world = [initial_position[:2].tolist(), list(endpoint), initial_position[:2].tolist()]
        else:
            waypoints_world = navigation.transform_body_waypoints(
                initial_position[:2], initial_yaw, navigation_config["body_waypoints"]
            )
        trajectory_carrier = None
        if args.execution_mode == "trajectory_carrier":
            if external_waypoints is not None:
                start_xy = np.asarray(external_waypoints[0], dtype=np.float64)
                end_xy = np.asarray(external_waypoints[-1], dtype=np.float64)
                direction_xy = end_xy - start_xy
                length = float(np.linalg.norm(direction_xy))
                if length <= 1.0e-6:
                    raise RuntimeError("trajectory carrier external route has zero length")
                lateral_errors = [
                    abs(float(np.cross(direction_xy, np.asarray(point, dtype=np.float64) - start_xy)))
                    / length
                    for point in external_waypoints
                ]
                if max(lateral_errors) > 1.0e-3:
                    raise RuntimeError(
                        "trajectory carrier requires a straight external route"
                    )
                carrier_endpoint_xy = end_xy.tolist()
            elif road_route_probe is not None and road_route_probe.get("accepted"):
                carrier_endpoint_xy = list(road_route_probe["selected_endpoint_world_xy"])
            else:
                raise RuntimeError(
                    "trajectory carrier requires an admitted PhysX road corridor or straight supported external route"
                )
            if args.safe_shuttle:
                raise ValueError("trajectory_carrier has its own smooth out-and-back cycle; do not use --safe-shuttle")
            trajectory_carrier = navigation.TrajectoryCarrier(
                start_world_xy=initial_position[:2].tolist(),
                end_world_xy=carrier_endpoint_xy,
                speed=(
                    float(args.trajectory_carrier_speed)
                    if args.trajectory_carrier_speed is not None
                    else float(
                        navigation_config.get(
                            "trajectory_carrier_speed",
                            navigation_config.get("max_forward_speed", 0.30),
                        )
                    )
                ),
                settle_duration_s=float(navigation_config.get("settle_duration_s", 1.0)),
                endpoint_hold_duration_s=float(navigation_config.get("trajectory_carrier_endpoint_hold_s", 0.6)),
                turn_duration_s=float(navigation_config.get("trajectory_carrier_turn_duration_s", 2.0)),
                turn_sign=float(args.turn_sign),
            )
            waypoints_world = trajectory_carrier.waypoints_world_xy
            waypoint_follower = None
        elif safe_shuttle_plan is not None:
            waypoint_follower = navigation.SafeShuttleFollower(
                legs=safe_shuttle_plan["legs"],
                max_forward_speed=float(navigation_config.get("max_forward_speed", 0.35)),
                max_reverse_speed=float(navigation_config.get("max_reverse_speed", 0.24)),
                max_yaw_rate=float(navigation_config.get("max_yaw_rate", 0.55)),
                heading_gain=float(navigation_config.get("heading_gain", 1.25)),
                heading_tolerance_rad=float(navigation_config.get("shuttle_heading_tolerance_rad", 0.12)),
                waypoint_tolerance=float(navigation_config.get("waypoint_tolerance", 0.30)),
                settle_duration_s=float(navigation_config.get("settle_duration_s", 1.0)),
                endpoint_stop_duration_s=float(navigation_config.get("loop_stop_duration_s", 0.8)),
            )
        elif navigation_config.get("tracking_controller") == "pure_pursuit":
            pure_pursuit = navigation_config.get("pure_pursuit", {})
            waypoint_follower = navigation.PurePursuitController(
                waypoints_world_xy=waypoints_world,
                max_forward_speed=float(
                    pure_pursuit.get("max_forward_speed", navigation_config.get("max_forward_speed", 0.34))
                ),
                max_tracking_yaw_rate=float(
                    pure_pursuit.get("max_tracking_yaw_rate", navigation_config.get("max_tracking_yaw_rate", 0.15))
                ),
                minimum_tracking_speed=float(
                    pure_pursuit.get("minimum_tracking_speed", navigation_config.get("minimum_tracking_speed", 0.22))
                ),
                lookahead_distance=float(pure_pursuit.get("lookahead_distance_m", 0.80)),
                curvature_speed_gain=float(pure_pursuit.get("curvature_speed_gain", 2.0)),
                vx_rate_limit=float(pure_pursuit.get("vx_rate_limit", 1.2)),
                yaw_rate_limit=float(pure_pursuit.get("yaw_rate_limit", 0.8)),
                goal_tolerance=float(pure_pursuit.get("goal_tolerance", 0.30)),
                deceleration_distance=float(pure_pursuit.get("deceleration_distance", 0.60)),
                settle_duration_s=float(navigation_config.get("settle_duration_s", 1.0)),
                stop_at_final_waypoint=bool(navigation_config.get("stop_at_final_waypoint", True)),
            )
        else:
            waypoint_follower = navigation.WaypointFollower(
                waypoints_world_xy=waypoints_world,
                max_forward_speed=float(navigation_config.get("max_forward_speed", 0.35)),
                max_yaw_rate=float(navigation_config.get("max_yaw_rate", 0.55)),
                max_tracking_yaw_rate=float(navigation_config.get("max_tracking_yaw_rate", 0.15)),
                minimum_tracking_speed=float(navigation_config.get("minimum_tracking_speed", 0.08)),
                turn_forward_speed=float(navigation_config.get("turn_forward_speed", 0.0)),
                heading_gain=float(navigation_config.get("heading_gain", 1.25)),
                turn_in_place_threshold_rad=float(
                    navigation_config.get("turn_in_place_threshold_rad", 0.18)
                ),
                waypoint_tolerance=float(navigation_config.get("waypoint_tolerance", 0.30)),
                settle_duration_s=float(navigation_config.get("settle_duration_s", 1.0)),
                loop_stop_duration_s=float(navigation_config.get("loop_stop_duration_s", 0.8)),
                stop_at_final_waypoint=bool(navigation_config.get("stop_at_final_waypoint", False)),
            )
        stuck_monitor = navigation.StuckMonitor(
            window_s=float(navigation_config.get("stuck_window_s", 3.0)),
            command_speed_threshold=float(navigation_config.get("stuck_command_speed", 0.15)),
            minimum_progress=float(navigation_config.get("stuck_minimum_progress", 0.08)),
            command_yaw_rate_threshold=float(navigation_config.get("stuck_command_yaw_rate", 0.20)),
            minimum_yaw_progress=float(navigation_config.get("stuck_minimum_yaw_progress_rad", 0.15)),
            confirmation_s=float(navigation_config.get("stuck_confirmation_s", 20.0)),
        )
        sustained_collision_monitor = navigation.SustainedCollisionMonitor(
            duration_s=float(navigation_config.get("sustained_collision_abort_s", 0.5))
        )
        post_goal_monitor = navigation.PostGoalStopMonitor(
            hold_s=float(navigation_config.get("post_goal_hold_s", 3.0))
        )

        stage = omni.usd.get_context().get_stage()
        robot_base_prim_path = "/World/envs/env_0/Robot/base"
        robot_base_prim = stage.GetPrimAtPath(robot_base_prim_path)
        if not robot_base_prim or not robot_base_prim.IsValid():
            raise RuntimeError(f"Go2 base prim not found: {robot_base_prim_path}")
        if args.coda_delivery:
            import omni.replicator.core as rep
            from isaacsim.sensors.rtx import LidarRtx

            epoch_ns = int(args.coda_epoch_ns if args.coda_epoch_ns is not None else time.time_ns())
            coda_writer = CodaDatasetWriter(
                run_dir / "coda_dataset",
                args.coda_sequence_name,
                epoch_ns,
                args.capture_fps,
                args.capture_fps,
            )
            coda_lidar = LidarRtx(
                prim_path=f"{robot_base_prim_path}/coda_lidar",
                name="coda_rtx_lidar",
                translation=np.asarray(args.rtx_lidar_mount_xyz, dtype=np.float64),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                config_file_name=args.rtx_lidar_config,
            )
            coda_lidar.initialize()
            coda_lidar_annotator = rep.AnnotatorRegistry.get_annotator(
                "RtxSensorCpuIsaacReadRTXLidarData"
            )
            coda_lidar_annotator.attach([coda_lidar.get_render_product_path()])
            coda_lidar_accumulator = RtxLidarAccumulator(epoch_ns)
        for name, definition in camera_definitions.items():
            sensor = unwrapped.scene[definition["scene_key"]]
            camera_prim = sensor._sensor_prims[0].GetPrim() if sensor._sensor_prims else None
            prim_path = str(camera_prim.GetPath()) if camera_prim and camera_prim.IsValid() else str(sensor.cfg.prim_path)
            if not camera_prim or not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Isaac Lab camera prim not found: {prim_path}")
            camera_rows.append(
                {
                    "name": name,
                    "sensor": sensor,
                    "camera_prim": camera_prim,
                    "calibration": definition["calibration"],
                    "render_calibration": definition["render_calibration"],
                }
            )
        if coda_writer is not None:
            coda_calibrations: dict[str, dict[str, Any]] = {}
            coda_transforms: dict[str, np.ndarray] = {}
            for name, definition in camera_definitions.items():
                if definition.get("calibration") is None:
                    continue
                role = str(definition["calibration"]["role"])
                delivery_name = str(
                    definition["calibration"].get("delivery_name", coda_camera_names[role])
                )
                render_calibration = definition["render_calibration"]
                coda_calibrations[delivery_name] = {
                    "K": render_calibration["K"],
                    "D": render_calibration["D"],
                    "camera_model": render_calibration.get(
                        "camera_model",
                        "pinhole" if role == "center_forward_pinhole" else "fisheye",
                    ),
                    "distortion_model": render_calibration.get(
                        "distortion_model",
                        "opencv_radtan" if role == "center_forward_pinhole" else "opencv_fisheye",
                    ),
                    "image_width": args.sensor_width,
                    "image_height": args.sensor_height,
                }
                coda_transforms[delivery_name] = coda_body_camera_transform(definition)
            body_lidar_transform = np.eye(4, dtype=np.float64)
            body_lidar_transform[:3, 3] = np.asarray(args.rtx_lidar_mount_xyz, dtype=np.float64)
            coda_writer.write_calibration(coda_calibrations, coda_transforms, body_lidar_transform)
        if args.strict_document_calibration:
            for row in camera_rows:
                if row["name"] == "overview":
                    continue
                map_x, map_y, valid, mapping_info = strict_calibration.build_inverse_map(
                    row["render_calibration"], args.sensor_width, args.sensor_height
                )
                strict_maps[row["name"]] = (map_x, map_y, valid)
                strict_mapping_metadata[row["name"]] = {
                    **mapping_info,
                    "K": row["render_calibration"]["K"],
                    "D": row["render_calibration"]["D"],
                    "image_size": row["render_calibration"]["image_size"],
                    "document_original_K": row["calibration"]["K"],
                    "document_original_D": row["calibration"]["D"],
                    "document_original_image_size": row["calibration"]["image_size"],
                    "intrinsics_scaled_for_render_resolution": (
                        row["render_calibration"]["image_size"] != row["calibration"]["image_size"]
                    ),
                    "camera_ext": strict_calibration.camera_ext(row["calibration"]),
                    "rgb_interpolation": "bilinear",
                    "depth_interpolation": "nearest",
                    "output_invalid_depth": "positive infinity",
                }
        settings.set("/rtx/post/tonemap/exposure", float(args.collection_exposure_ev))
        appearance_application: dict[str, Any] = {
            "id": "baseline",
            "application": "source appearance with collection exposure override",
            "seed": args.seed,
            "exposure_ev": float(args.collection_exposure_ev),
        }
        if args.appearance_profile != "baseline":
            from pxr import Sdf

            variant = next(item for item in appearance.VARIANTS if item["id"] == args.appearance_profile)
            if variant.get("kind") == "camera":
                appearance_application = dict(pre_camera_application or {})
                settings.set("/rtx/post/tonemap/exposure", float(appearance_application["exposure_ev"]))
                central_camera = UsdGeom.Camera(camera_rows[0]["camera_prim"])
                appearance_application.update(
                    {
                        "actual_focal_length": float(central_camera.GetFocalLengthAttr().Get()),
                        "actual_horizontal_aperture": float(central_camera.GetHorizontalApertureAttr().Get()),
                        "actual_vertical_aperture": float(central_camera.GetVerticalApertureAttr().Get()),
                        "mounts": {
                            row["name"]: {
                                "position_vehicle_xyz_m": camera_definitions[row["name"]]["mount_position_vehicle_xyz_m"],
                                "yaw_rad": camera_definitions[row["name"]]["mount_yaw_rad"],
                            }
                            for row in camera_rows
                            if row["name"] != "overview"
                        },
                    }
                )
            else:
                variant_layer = Sdf.Layer.CreateAnonymous("urbanverse_go2_route_appearance.usda")
                stage.GetSessionLayer().subLayerPaths.append(variant_layer.identifier)
                stage.SetEditTarget(variant_layer)
                central = camera_rows[0]
                central_camera = UsdGeom.Camera(central["camera_prim"])
                # Non-camera appearance variants require a writable transform
                # op in their shared helper. Keep it on an isolated session
                # camera so the articulated CameraCfg mount is untouched.
                dummy = UsdGeom.Camera.Define(stage, "/UrbanVerseEvaluation/VariantParameterCamera")
                dummy_xform = UsdGeom.Xformable(dummy.GetPrim())
                dummy_xform.ClearXformOpOrder()
                dummy_op = dummy_xform.AddTransformOp()
                base_matrix = world_camera_matrix(initial_position, initial_quat, central["calibration"])
                appearance_application = appearance.apply_variant(
                    stage,
                    variant,
                    dummy,
                    dummy_op,
                    base_matrix,
                    float(central_camera.GetFocalLengthAttr().Get()),
                    float(central_camera.GetHorizontalApertureAttr().Get()),
                    float(central_camera.GetVerticalApertureAttr().Get()),
                    float(source_stage_metadata["meters_per_unit"]),
                    args.seed,
                )
        phase1_vehicle_actor = None
        phase1_yield_controller = None
        phase1_vehicle_state = None
        phase1_avoidance_diagnostics = None
        phase1_overview_view = None
        if args.phase1_dynamic_vehicle:
            route_start = np.asarray(waypoints_world[0], dtype=np.float64)
            route_end = np.asarray(waypoints_world[-1], dtype=np.float64)
            route_vector = route_end - route_start
            route_length = float(np.linalg.norm(route_vector))
            if route_length < 2.0:
                raise RuntimeError(f"phase-one route is too short for a controlled crossing: {route_length:.3f}")
            # Keep the initial Go2 stance outside the SUV's full swept-width
            # corridor, leaving enough road for a policy-executed stop before
            # the vehicle reaches the near-endpoint conflict zone.
            conflict_xy = route_start + 0.96 * route_vector
            crossing_schedule = phase1_dynamic.CrossingSchedule(
                conflict_xy=conflict_xy,
                # The selected SUV's long axis is +X in its authored root.  A
                # +X crossing intersects this diagonal Go2 road at about 60°
                # without introducing an extra root-orientation calibration.
                direction_xy=np.asarray([1.0, 0.0], dtype=np.float64),
                distance=float(args.phase1_vehicle_distance),
                crossing_time_s=float(args.phase1_crossing_time_s),
                speed=float(args.phase1_vehicle_speed),
            )
            phase1_vehicle_actor = phase1_dynamic.OriginalSceneVehicleActor(
                stage, args.phase1_vehicle_prim, crossing_schedule
            )
            phase1_vehicle_state = phase1_vehicle_actor.update(0.0)
            phase1_overview_view = phase1_overview_camera_view(route_start, conflict_xy)
            phase1_yield_controller = phase1_dynamic.Go2YieldController(
                enabled=args.phase1_avoidance_mode == "yield",
                dt=dt,
            )
            for _ in range(4):
                simulation_app.update()
            print(
                "PHASE1_VEHICLE_GEOMETRY "
                + json.dumps(phase1_vehicle_actor.measure_world_geometry(), ensure_ascii=False),
                flush=True,
            )
            write_json(
                metadata / "phase1_dynamic_vehicle_config.json",
                {
                    "stage": "phase_one_stop_yield_resume",
                    "avoidance_mode": args.phase1_avoidance_mode,
                    "vehicle": phase1_vehicle_actor.summary(),
                    "go2_controller": phase1_yield_controller.summary(),
                    "responsibility_contract": {
                        "vehicle_observes_go2": False,
                        "vehicle_schedule_input": "simulation timestamp only",
                        "go2_observes_vehicle": True,
                        "go2_avoidance_output": "velocity scale applied before frozen locomotion policy",
                        "go2_base_pose_prescribed": False,
                    },
                },
            )
        if args.scene10_continuous_traffic:
            traffic_manager = Scene10ContinuousVehicleManager(
                stage,
                scene_query,
                args.traffic_registry.resolve(),
                (
                    args.traffic_audit_inventory.resolve()
                    if args.traffic_audit_inventory is not None
                    else None
                ),
                dt=dt,
                validated_routes_path=(
                    args.traffic_validated_routes.resolve()
                    if args.traffic_validated_routes is not None
                    else None
                ),
                automotive_routes_path=args.traffic_automotive_routes.resolve(),
                random_seed=args.seed,
                traffic_vehicle_count=args.traffic_vehicle_count,
                initial_fill=(
                    args.traffic_initial_fill
                    or bool(
                        portable_scene is not None
                        and portable_scene.traffic.get("initial_fill", False)
                    )
                ),
                keep_vehicle_prims_active=True,
                fabric_xform_views=None,
                use_source_vehicle_prims=bool(preauthored_traffic_sources),
                preauthored_vehicle_prim_paths=[
                    row["controlled_prim"] for row in preauthored_traffic_sources
                ],
                scene_config_path=(
                    args.traffic_scene_config.resolve()
                    if args.traffic_scene_config is not None
                    else None
                ),
            )
            initial_traffic_state = traffic_manager.update(
                0.0,
                initial_position[:2],
                initial_yaw,
            )
            traffic_manager.refresh_visual_transforms()
            write_json(
                metadata / "scene10_continuous_traffic_config.json",
                {
                    "enabled": True,
                    "registry": str(args.traffic_registry.resolve()),
                    "registry_sha256": sha256(args.traffic_registry.resolve()),
                    "scene_config": (
                        str(args.traffic_scene_config.resolve())
                        if args.traffic_scene_config is not None
                        else None
                    ),
                    "audit_inventory": (
                        str(args.traffic_audit_inventory.resolve())
                        if args.traffic_audit_inventory is not None
                        else None
                    ),
                    "audit_inventory_sha256": (
                        sha256(args.traffic_audit_inventory.resolve())
                        if args.traffic_audit_inventory is not None
                        else None
                    ),
                    "validated_routes": (
                        str(args.traffic_validated_routes.resolve())
                        if args.traffic_validated_routes is not None
                        else None
                    ),
                    "validated_routes_sha256": (
                        sha256(args.traffic_validated_routes.resolve())
                        if args.traffic_validated_routes is not None
                        else None
                    ),
                    "automotive_routes": str(args.traffic_automotive_routes.resolve()),
                    "automotive_routes_sha256": sha256(args.traffic_automotive_routes.resolve()),
                    "preauthored_sources": preauthored_traffic_sources,
                    "traffic_vehicle_count": args.traffic_vehicle_count,
                    "initial_fill": args.traffic_initial_fill,
                    "go2_runtime_support": go2_runtime_support,
                    "random_seed": args.seed,
                    "minimum_go2_vehicle_clearance_m": args.minimum_go2_vehicle_clearance,
                    "maximum_visual_center_error_m": args.maximum_traffic_visual_center_error,
                    "maximum_rendered_body_axis_error_deg": args.maximum_rendered_body_axis_error,
                    "minimum_visible_vehicle_pass_duration_s": args.minimum_visible_vehicle_pass_duration,
                    "initial_state": initial_traffic_state,
                    "go2_interaction": "none; spatially separated NearRoad route",
                },
            )
        if pedestrian_config is not None:
            people_manager = OfficialPeopleManager(stage, pedestrian_config, run_dir)
            write_json(
                metadata / "animated_pedestrian_config.json",
                {
                    "enabled": True,
                    "source": str(portable_scene.pedestrian_config.resolve()),
                    "source_sha256": sha256(portable_scene.pedestrian_config.resolve()),
                    "preauthored_people": preauthored_people,
                },
            )
        reference = [
            {
                "waypoint_index": index,
                "position": [float(point[0]), float(point[1]), float(initial_position[2])],
            }
            for index, point in enumerate(waypoints_world)
        ]
        write_json(
            metadata / "control_schedule.json",
            {
                "mode": route_mode,
                "execution_mode": (
                    "trajectory_carrier_proxy"
                    if trajectory_carrier is not None
                    else "safe_shuttle"
                    if safe_shuttle_plan is not None
                    else "waypoint_follower"
                ),
                "execution_disclosure": (
                    "The articulated Go2 base pose is prescribed on a PhysX-prevalidated road. The frozen policy animates joints only; contacts, gait and obstacle traversal are not validated."
                    if trajectory_carrier is not None
                    else "The frozen policy controls the articulated Go2 base through physics."
                ),
                "step_dt_s": dt,
                "navigation": navigation_config,
                "duration_s": total_duration,
                "phase1_dynamic_vehicle": (
                    {
                        "enabled": True,
                        "avoidance_mode": args.phase1_avoidance_mode,
                        "vehicle_prim": args.phase1_vehicle_prim,
                        "config": str(metadata / "phase1_dynamic_vehicle_config.json"),
                    }
                    if phase1_vehicle_actor is not None
                    else None
                ),
                "scene10_continuous_traffic": (
                    {
                        "enabled": True,
                        "config": str(metadata / "scene10_continuous_traffic_config.json"),
                        "lifecycle": (
                            "random cooldown -> safe spawn -> forward bicycle drive -> "
                            "endpoint disappearance -> random cooldown"
                        ),
                        "go2_interaction": "none; spatially separated NearRoad route",
                    }
                    if traffic_manager is not None
                    else None
                ),
                "legacy_open_loop_schedule_retained_for_provenance_only": schedule,
            },
        )
        write_json(
            metadata / "reference_route.json",
            {
                "records": reference,
                "origin": (
                    "externally supplied scene-matched route with prior static PhysX occupancy validation"
                    if external_waypoints is not None
                    else
                    "smooth out-and-back carrier path on a pre-motion PhysX-admitted road corridor"
                    if trajectory_carrier is not None
                    else
                    "A* + ESDF + spline global reference route, densely PhysX re-validated over the robot-width corridor before motion"
                    if global_route_plan is not None
                    else
                    "forward-only out-and-back road spokes selected by a pre-motion PhysX road-corridor probe"
                    if safe_shuttle_plan is not None
                    else "closed-loop out-and-back waypoints selected by a pre-motion PhysX road-corridor probe"
                    if road_route_probe is not None
                    else "closed-loop waypoints transformed from the actual Go2 base pose after PhysX ground probing"
                ),
                "coordinate_definition": "world XY route admitted once before motion; commands then use live pose feedback",
                "road_route_probe": str(metadata / "road_route_probe.json") if road_route_probe is not None else None,
                "global_route_plan": str(metadata / "global_route_plan.json") if global_route_plan is not None else None,
                "safe_shuttle_plan": str(metadata / "safe_shuttle_plan.json") if safe_shuttle_plan is not None else None,
            },
        )
        for row in camera_rows:
            if row["name"] != "overview":
                definition = camera_definitions[row["name"]]
                row["mount_local_transform"] = {
                    "position_vehicle_xyz_m": list(definition["mount_position_vehicle_xyz_m"]),
                    "mount_yaw_rad": float(definition["mount_yaw_rad"]),
                    "mount_quaternion_wxyz": list(definition["mount_quaternion_wxyz"]),
                    "orientation_convention": definition["mount_convention"],
                    "orientation_application": definition["orientation_application"],
                    "sensor_update_period_s": definition["update_period_s"],
                    "document_camera_ext": definition["document_camera_ext"],
                }
        eye, target = (
            phase1_overview_view
            if phase1_overview_view is not None
            else overview_camera_view(
                initial_position,
                initial_quat,
                args.overview_chase_distance,
                args.overview_chase_lateral_offset,
                args.overview_chase_height,
                args.overview_target_forward,
                args.overview_target_height,
            )
        )
        overview_sensor = next(row["sensor"] for row in camera_rows if row["name"] == "overview")
        overview_sensor.set_world_poses_from_view(
            torch.tensor([eye.tolist()], device=unwrapped.device, dtype=torch.float32),
            torch.tensor([target.tolist()], device=unwrapped.device, dtype=torch.float32),
        )
        for _ in range(args.warmup_updates):
            simulation_app.update()

        trajectory_path = metadata / "trajectory.jsonl"
        frame_index_path = metadata / "frame_index.jsonl"
        trajectory_handle = trajectory_path.open("w", encoding="utf-8")
        frame_handle = frame_index_path.open("w", encoding="utf-8")
        frame_count = 0
        overview_frames: list[np.ndarray] = []
        review_frame_paths: list[Path] = []
        actual_xy: list[list[float]] = []
        step_times: list[float] = []
        collision_steps = 0
        foot_support_steps = 0
        nonfinite_steps = 0
        route_stuck = False
        stuck_event: dict[str, Any] | None = None
        route_collision_abort = False
        collision_event: dict[str, Any] | None = None
        post_goal_stop = False
        post_goal_state: dict[str, Any] = post_goal_monitor.update(0.0, False)
        maximum_distance_from_start = 0.0
        maximum_carrier_xy_error = 0.0
        maximum_carrier_yaw_error = 0.0
        reference_xy = [row["position"][:2] for row in reference]
        traffic_state: dict[str, Any] | None = None
        capture_loop_started = time.perf_counter()
        for step_index in range(total_steps):
            timestamp_before = step_index * dt
            position_before = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            quaternion_before = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            if traffic_manager is not None:
                traffic_state = traffic_manager.update(
                    timestamp_before,
                    position_before[:2],
                    yaw_from_quat(quaternion_before),
                )
            if people_manager is not None:
                people_manager.prepare_step(
                    traffic_manager.agents if traffic_manager is not None else None,
                    position_before[:2],
                    robot.data.root_lin_vel_w[0, :2]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64),
                )
            carrier_sample = None
            if trajectory_carrier is not None:
                carrier_sample = trajectory_carrier.sample((step_index + 1) * dt)
                command = carrier_sample
            else:
                command = waypoint_follower.command(
                    position_before[:2], yaw_from_quat(quaternion_before), timestamp_before
                )
            base_route_command = copy.deepcopy(command)
            if phase1_vehicle_actor is not None:
                phase1_vehicle_state = phase1_vehicle_actor.update(timestamp_before)
                command, phase1_avoidance_diagnostics = phase1_yield_controller.apply(
                    base_route_command,
                    timestamp_s=timestamp_before,
                    go2_position_xy=position_before[:2],
                    go2_yaw_rad=yaw_from_quat(quaternion_before),
                    vehicle_state=phase1_vehicle_state,
                    vehicle_half_extents_xy=phase1_vehicle_actor.half_extents_xy,
                )
            schedule_index = int(command["waypoint_index"])
            command_tensor = torch.tensor(
                [[float(command["linear_x"]), float(command["linear_y"]), float(command["angular_z"])]],
                device=unwrapped.device,
                dtype=torch.float32,
            )
            command_term.vel_command_b[:] = command_tensor
            command_term.time_left[:] = 1000.0
            command_term.is_heading_env[:] = False
            command_term.is_standing_env[:] = False
            observation_tensor = observations["policy"] if isinstance(observations, dict) else observations
            before_step = time.perf_counter()
            with torch.inference_mode():
                actions = policy(observation_tensor)
                observations, rewards, terminated, truncated, info = env.step(actions)
            if coda_lidar_annotator is not None and coda_lidar_accumulator is not None:
                raw_lidar_tick = coda_lidar_annotator.get_data()
                if isinstance(raw_lidar_tick, dict) and len(raw_lidar_tick.get("distances", [])) > 0:
                    coda_lidar_accumulator.push(raw_lidar_tick)
            if people_manager is not None:
                people_manager.sample()
            if carrier_sample is not None:
                target_xy = carrier_sample["target_position_world_xy"]
                target_yaw = float(carrier_sample["target_yaw_rad_world"])
                target_pose = torch.tensor(
                    [[
                        float(target_xy[0]),
                        float(target_xy[1]),
                        float(initial_position[2] + args.trajectory_carrier_height_offset_m),
                        math.cos(target_yaw / 2.0),
                        0.0,
                        0.0,
                        math.sin(target_yaw / 2.0),
                    ]],
                    dtype=robot.data.root_pos_w.dtype,
                    device=robot.data.root_pos_w.device,
                )
                world_xy_velocity = carrier_sample["target_linear_velocity_world_xy"]
                target_velocity = torch.tensor(
                    [[
                        float(world_xy_velocity[0]),
                        float(world_xy_velocity[1]),
                        0.0,
                        0.0,
                        0.0,
                        float(carrier_sample["angular_z"]),
                    ]],
                    dtype=robot.data.root_lin_vel_w.dtype,
                    device=robot.data.root_lin_vel_w.device,
                )
                # ``env.step`` refreshes Isaac Lab's root buffers while
                # inference mode is active.  Updating those buffers must stay
                # in the same mode as well.
                with torch.inference_mode():
                    robot.write_root_pose_to_sim(target_pose)
                    robot.write_root_velocity_to_sim(target_velocity)
            step_times.append(time.perf_counter() - before_step)
            timestamp = (step_index + 1) * dt
            position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            quaternion = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            linear_velocity = robot.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
            angular_velocity = robot.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float64)
            joint_position = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
            joint_velocity = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float64)
            contact = contact_sensor.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
            contact_norms = np.linalg.norm(contact, axis=-1)
            contact_classification = navigation.classify_contacts(contact_sensor.body_names, contact_norms)
            body_collision = bool(contact_classification["non_foot_collision"])
            collision_steps += int(body_collision)
            foot_support_steps += int(bool(contact_classification["supporting_feet"]))
            sustained_collision_state = sustained_collision_monitor.update(timestamp, body_collision)
            if (
                trajectory_carrier is None
                and sustained_collision_state["sustained"]
                and not route_collision_abort
            ):
                route_collision_abort = True
                collision_event = {
                    "step_index": step_index + 1,
                    "timestamp_s": timestamp,
                    "position_world": position.tolist(),
                    "command": command,
                    "contact_classification": contact_classification,
                    "detection": sustained_collision_state,
                }
            if trajectory_carrier is None:
                stuck_state = stuck_monitor.update(
                    timestamp,
                    position[:2],
                    float(command["linear_x"]),
                    yaw_from_quat(quaternion),
                    float(command["angular_z"]),
                )
            else:
                carrier_xy_error = float(
                    np.linalg.norm(position[:2] - np.asarray(command["target_position_world_xy"], dtype=np.float64))
                )
                carrier_yaw_error = abs(
                    navigation.wrap_angle(yaw_from_quat(quaternion) - float(command["target_yaw_rad_world"]))
                )
                maximum_carrier_xy_error = max(maximum_carrier_xy_error, carrier_xy_error)
                maximum_carrier_yaw_error = max(maximum_carrier_yaw_error, carrier_yaw_error)
                stuck_state = {
                    "stuck": False,
                    "applicable": False,
                    "reason": "trajectory carrier pose is prescribed; physical-progress monitoring is not applicable",
                    "carrier_xy_error": carrier_xy_error,
                    "carrier_yaw_error_rad": carrier_yaw_error,
                }
            if stuck_state["stuck"] and not route_stuck:
                route_stuck = True
                stuck_event = {
                    "step_index": step_index + 1,
                    "timestamp_s": timestamp,
                    "position_world": position.tolist(),
                    "command": command,
                    "detection": stuck_state,
                }
            finite_pose = bool(np.isfinite(position).all() and np.isfinite(quaternion).all())
            nonfinite_steps += int(not finite_pose)
            post_goal_state = post_goal_monitor.update(
                timestamp,
                bool(
                    trajectory_carrier is None
                    and waypoint_follower.stop_at_final_waypoint
                    and waypoint_follower.route_complete
                ),
            )
            post_goal_stop = bool(post_goal_state["stop_due"])
            actual_xy.append(position[:2].tolist())
            maximum_distance_from_start = max(
                maximum_distance_from_start,
                float(np.linalg.norm(position[:2] - initial_position[:2])),
            )
            trajectory_record = {
                "step_index": step_index + 1,
                "timestamp_s": timestamp,
                "schedule_index": schedule_index,
                "schedule_label": command["label"],
                "tracking_debug": command.get("tracking_debug"),
                "base_route_command_before_dynamic_avoidance": base_route_command,
                "phase1_dynamic_vehicle": (
                    phase1_vehicle_actor.serializable_state()
                    if phase1_vehicle_actor is not None
                    else None
                ),
                "phase1_go2_avoidance": phase1_avoidance_diagnostics,
                "scene10_continuous_traffic": traffic_state,
                "control_command_body": command_tensor[0].detach().cpu().numpy().tolist(),
                "action_joint_position_policy": actions[0].detach().cpu().numpy().tolist(),
                "base_position_world": position.tolist(),
                "base_quaternion_wxyz_world": quaternion.tolist(),
                "base_yaw_rad_world": yaw_from_quat(quaternion),
                "base_linear_velocity_world": linear_velocity.tolist(),
                "base_angular_velocity_world": angular_velocity.tolist(),
                "joint_position": joint_position.tolist(),
                "joint_velocity": joint_velocity.tolist(),
                "contact_body_names": list(contact_sensor.body_names),
                "contact_force_norms": contact_norms.tolist(),
                "body_collision": body_collision,
                "contact_classification": contact_classification,
                "contact_evidence_valid": trajectory_carrier is None,
                "sustained_collision_detection": sustained_collision_state,
                "stuck_detection": stuck_state,
                "execution_mode": args.execution_mode,
                "prescribed_carrier_target": carrier_sample,
                "terminated": bool(terminated[0].item()),
                "truncated": bool(truncated[0].item()),
                "reward": float(rewards[0].item()),
                "finite_pose": finite_pose,
            }
            append_jsonl(trajectory_handle, trajectory_record)

            if (
                step_index % capture_stride != 0
                and step_index != total_steps - 1
                and not route_stuck
                and not route_collision_abort
                and not post_goal_stop
            ):
                continue
            if phase1_vehicle_actor is not None:
                phase1_vehicle_actor.refresh_visual_transform()
            if traffic_manager is not None:
                traffic_manager.refresh_visual_transforms()
            if people_manager is not None:
                people_manager.refresh_visual_transforms()
            camera_world = {}
            for row in camera_rows:
                if row["name"] == "overview":
                    eye, target = (
                        phase1_overview_view
                        if phase1_overview_view is not None
                        else overview_camera_view(
                            position,
                            quaternion,
                            args.overview_chase_distance,
                            args.overview_chase_lateral_offset,
                            args.overview_chase_height,
                            args.overview_target_forward,
                            args.overview_target_height,
                        )
                    )
                    row["sensor"].set_world_poses_from_view(
                        torch.tensor([eye.tolist()], device=unwrapped.device, dtype=torch.float32),
                        torch.tensor([target.tolist()], device=unwrapped.device, dtype=torch.float32),
                    )
                    matrix = overview_camera_matrix_from_view(eye, target)
                else:
                    matrix = world_camera_matrix(position, quaternion, row["calibration"])
                camera_world[row["name"]] = matrix_rows(matrix)
            simulation_app.update()
            if coda_lidar_annotator is not None and coda_lidar_accumulator is not None:
                raw_lidar_tick = coda_lidar_annotator.get_data()
                if isinstance(raw_lidar_tick, dict) and len(raw_lidar_tick.get("distances", [])) > 0:
                    coda_lidar_accumulator.push(raw_lidar_tick)
            for row in camera_rows:
                row["sensor"].update(0.0, force_recompute=True)
            camera_pose_readback = {
                row["name"]: {
                    "position_world": row["sensor"].data.pos_w[0].detach().cpu().numpy().tolist(),
                    "quaternion_wxyz_world_convention": row["sensor"]
                    .data.quat_w_world[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "quaternion_wxyz_ros_convention": row["sensor"]
                    .data.quat_w_ros[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                }
                for row in camera_rows
            }
            frame_record: dict[str, Any] = {
                "frame_index": frame_count,
                "step_index": step_index + 1,
                "timestamp_s": timestamp,
                "schedule_index": schedule_index,
                "schedule_label": command["label"],
                "phase1_dynamic_vehicle": (
                    phase1_vehicle_actor.serializable_state()
                    if phase1_vehicle_actor is not None
                    else None
                ),
                "phase1_go2_avoidance": phase1_avoidance_diagnostics,
                "scene10_continuous_traffic": traffic_state,
                "body_collision": body_collision,
                "contact_classification": contact_classification,
                "contact_evidence_valid": trajectory_carrier is None,
                "sustained_collision_detection": sustained_collision_state,
                "stuck_detection": stuck_state,
                "execution_mode": args.execution_mode,
                "prescribed_carrier_target": carrier_sample,
                "base_position_world": position.tolist(),
                "base_quaternion_wxyz_world": quaternion.tolist(),
                "sensor_rig_quaternion_wxyz_world": quaternion.tolist(),
                "camera_world_transforms": camera_world,
                "camera_pose_readback": camera_pose_readback,
                "modalities": {},
            }
            coda_scan = None
            if coda_writer is not None:
                if coda_lidar_accumulator is None or coda_lidar_accumulator.point_count <= 0:
                    raise RuntimeError(f"no RTX LiDAR returns accumulated for camera frame {frame_count}")
                # The RGB exposure and this raw RTX tick come from the same
                # render update.  Use the scan midpoint as their shared Isaac
                # sensor-clock timestamp; control-loop time excludes explicit
                # render updates and is not a valid synchronization clock.
                coda_scan = coda_lidar_accumulator.pop_latest_scan()
                coda_timestamp_ns = coda_scan.reference_timestamp_ns
            else:
                coda_timestamp_ns = None
            if coda_writer is not None and coda_timestamp_ns is not None:
                coda_writer.add_ego_pose(coda_timestamp_ns, position, quaternion)
            derived_keyframe = bool(
                frame_count % args.derived_visualization_stride == 0 or step_index == total_steps - 1
            )
            full_metrics_keyframe = bool(
                frame_count % args.full_metrics_stride == 0 or step_index == total_steps - 1
            )
            if args.lightweight_three_panel:
                centre = next(row for row in camera_rows if row["name"] != "overview")
                overview = next(row for row in camera_rows if row["name"] == "overview")
                depth = normalise_array(centre["sensor"].data.output["distance_to_camera"][0])
                rgb = normalise_rgb(overview["sensor"].data.output["rgb"][0])
                finite = np.isfinite(depth) & (depth > 0.0)
                overview_dir = captures / "third_person_rgb"
                review_dir = captures / "three_panel_review_frames"
                overview_dir.mkdir(parents=True, exist_ok=True)
                review_dir.mkdir(parents=True, exist_ok=True)
                overview_path = overview_dir / f"frame_{frame_count:04d}.png"
                review_path = review_dir / f"frame_{frame_count:04d}.png"
                Image.fromarray(rgb, mode="RGB").save(overview_path)
                review = lightweight_review_frame(
                    depth,
                    rgb,
                    reference_xy,
                    actual_xy,
                    timestamp,
                    actual_capture_fps,
                    args.playback_speed,
                )
                Image.fromarray(review, mode="RGB").save(review_path)
                review_frame_paths.append(review_path)
                frame_paths.append(review_path)
                frame_record["modalities"] = {
                    centre["name"]: {
                        "depth_rendered_for_review_only": True,
                        "depth_float32": None,
                        "finite_depth_ratio": float(np.mean(finite)),
                    },
                    "overview": {
                        "rgb": str(overview_path.relative_to(run_dir)),
                        "nonempty_rgb": bool(rgb.size and np.any(rgb)),
                    },
                    "review": {"rgb": str(review_path.relative_to(run_dir))},
                }
            else:
                for row in camera_rows:
                    name = row["name"]
                    output = row["sensor"].data.output
                    rgb = normalise_rgb(output["rgb"][0])
                    depth = normalise_array(output["distance_to_camera"][0])
                    if args.strict_document_calibration and name != "overview":
                        if frame_count == 0:
                            native_rgb_dir = captures / "native_projection_first_frame" / "rgb" / name
                            native_depth_dir = captures / "native_projection_first_frame" / "depth" / name
                            native_rgb_dir.mkdir(parents=True, exist_ok=True)
                            native_depth_dir.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(rgb, mode="RGB").save(native_rgb_dir / "frame_0000.png")
                            np.save(native_depth_dir / "frame_0000.npy", depth.astype(np.float32))
                        rgb, depth = strict_calibration.remap_rgb_depth(
                            rgb, depth, *strict_maps[name]
                        )
                    rgb_dir = captures / "rgb" / name
                    depth_dir = captures / "depth" / name
                    vis_dir = captures / "depth_visualization" / name
                    for directory in (rgb_dir, depth_dir, vis_dir):
                        directory.mkdir(parents=True, exist_ok=True)
                    rgb_path = rgb_dir / f"frame_{frame_count:04d}.png"
                    depth_path = depth_dir / f"frame_{frame_count:04d}.npy"
                    vis_path = vis_dir / f"frame_{frame_count:04d}.png"
                    Image.fromarray(rgb, mode="RGB").save(rgb_path)
                    np.save(depth_path, depth.astype(np.float32))
                    if coda_writer is not None and name != "overview" and coda_timestamp_ns is not None:
                        delivery_name = str(
                            row["calibration"].get(
                                "delivery_name",
                                coda_camera_names[str(row["calibration"]["role"])],
                            )
                        )
                        delivery_path = coda_writer.add_camera_frame(
                            delivery_name,
                            frame_count,
                            coda_timestamp_ns,
                            rgb,
                        )
                    else:
                        delivery_path = None
                    if derived_keyframe:
                        save_depth_visualization(depth, vis_path)
                    frame_record["modalities"][name] = {
                        "rgb": str(rgb_path.relative_to(run_dir)),
                        "depth_float32": str(depth_path.relative_to(run_dir)),
                        "depth_visualization": str(vis_path.relative_to(run_dir)) if derived_keyframe else None,
                        "depth_visualization_is_keyframe": derived_keyframe,
                        "rgb_metrics": navigation_frame_metrics(rgb, depth, full_metrics_keyframe),
                        "coda_rgb": (
                            str(delivery_path.relative_to(coda_writer.root))
                            if delivery_path is not None and coda_writer is not None
                            else None
                        ),
                    }
                    if name == "overview":
                        annotated = annotate_overview(
                            rgb,
                            args.scene,
                            frame_count,
                            timestamp,
                            command,
                            position.tolist(),
                            float(np.linalg.norm(linear_velocity[:2])),
                            body_collision,
                            route_stuck or route_collision_abort,
                            actual_xy,
                            reference_xy,
                            args.appearance_profile,
                            args.execution_mode,
                            trajectory_carrier is None,
                        )
                        annotated_dir = captures / "overview_annotated"
                        annotated_dir.mkdir(parents=True, exist_ok=True)
                        annotated_path = annotated_dir / f"frame_{frame_count:04d}.png"
                        Image.fromarray(annotated, mode="RGB").save(annotated_path)
                        overview_frames.append(annotated)
                        frame_paths.append(annotated_path)
            if not args.omit_motion_vectors and not args.lightweight_three_panel:
                central_output = camera_rows[0]["sensor"].data.output
                motion = normalise_array(central_output["motion_vectors"][0])
                motion_dir = captures / "motion_vectors" / camera_calibrations[0]["id"]
                motion_dir.mkdir(parents=True, exist_ok=True)
                motion_path = motion_dir / f"frame_{frame_count:04d}.npy"
                np.save(motion_path, motion.astype(np.float32))
                frame_record["modalities"][camera_calibrations[0]["id"]]["motion_vectors"] = str(
                    motion_path.relative_to(run_dir)
                )
            if coda_writer is not None and coda_scan is not None:
                coda_scan_path = coda_writer.add_lidar_scan(coda_lidar_scan_count, coda_scan)
                coda_lidar_scan_count += 1
                coda_camera_lidar_offsets_ns.append(
                    abs(int(coda_scan.reference_timestamp_ns) - int(coda_timestamp_ns))
                )
                frame_record["coda_lidar"] = {
                    "pcd": str(coda_scan_path.relative_to(coda_writer.root)),
                    "point_count": int(coda_scan.xyz.shape[0]),
                    "timestamp_start_ns": coda_scan.timestamp_start_ns,
                    "timestamp_end_ns": coda_scan.timestamp_end_ns,
                    "reference_timestamp_ns": coda_scan.reference_timestamp_ns,
                    "camera_reference_offset_ns": coda_camera_lidar_offsets_ns[-1],
                }
            append_jsonl(frame_handle, frame_record)
            frame_count += 1
            if route_stuck or route_collision_abort or post_goal_stop:
                event_label = (
                    "ROUTE_STUCK"
                    if route_stuck
                    else "ROUTE_COLLISION_ABORT"
                    if route_collision_abort
                    else "ROUTE_COMPLETE_CAPTURE_STOP"
                )
                event = (
                    stuck_event
                    if route_stuck
                    else collision_event
                    if route_collision_abort
                    else post_goal_state
                )
                safe_print(
                    event_label + " " + json.dumps(event, ensure_ascii=False, default=appearance.json_default),
                    flush=True,
                )
                break

        capture_loop_wall_s = time.perf_counter() - capture_loop_started
        trajectory_handle.close()
        frame_handle.close()
        coda_max_sync_offset_ns = (
            max(coda_camera_lidar_offsets_ns) if coda_camera_lidar_offsets_ns else None
        )
        if coda_writer is not None:
            coda_known_issues = []
            if coda_max_sync_offset_ns is not None and coda_max_sync_offset_ns > 10_000_000:
                coda_known_issues.append(
                    f"maximum camera-LiDAR reference offset {coda_max_sync_offset_ns} ns exceeds 10 ms"
                )
            coda_writer.finalize(coda_known_issues)
            coda_validation = validate_dataset(
                coda_writer.root,
                coda_writer.sequence_name,
                require_three_pinhole=args.coda_pinhole_rig,
            )
            write_json(metadata / "coda_validation.json", coda_validation)
            write_json(
                metadata / "coda_delivery.json",
                {
                    "dataset_root": str(coda_writer.root),
                    "sequence_name": coda_writer.sequence_name,
                    "epoch_ns": coda_writer.epoch_ns,
                    "camera_rig": {
                        "three_pinhole": bool(args.coda_pinhole_rig),
                        "horizontal_fov_deg": (
                            float(args.coda_pinhole_horizontal_fov_deg)
                            if args.coda_pinhole_rig
                            else None
                        ),
                        "side_yaw_deg": (
                            float(args.coda_pinhole_side_yaw_deg)
                            if args.coda_pinhole_rig
                            else None
                        ),
                        "body_frame_convention": "+X forward, +Y left, +Z up",
                        "simultaneous_render_timestamps": True,
                        "shared_render_and_appearance_settings": True,
                    },
                    "camera_frame_count_per_camera": frame_count,
                    "lidar_scan_count": coda_lidar_scan_count,
                    "maximum_camera_lidar_reference_offset_ns": coda_max_sync_offset_ns,
                    "lidar_config": args.rtx_lidar_config,
                    "lidar_mount_body_xyz_m": list(args.rtx_lidar_mount_xyz),
                    "point_coordinate_frame": "native RTX LiDAR frame",
                    "ring_source": "RTX channels; emitterIds fallback is rejected unless point-aligned",
                    "point_timestamp_source": "RTX timestampNs + per-return deltaTimes",
                },
            )
        video_path = videos / (
            "go2_route_three_panel_2x.webm" if args.lightweight_three_panel else "go2_route_overview.webm"
        )
        encoding_started = time.perf_counter()
        video_metadata = (
            encode_video_paths(review_frame_paths, video_path, actual_capture_fps * args.playback_speed)
            if args.lightweight_three_panel
            else encode_video(overview_frames, video_path, actual_capture_fps)
        )
        video_encoding_wall_s = time.perf_counter() - encoding_started
        contact_sheet = visualizations / "contact_sheet.png"
        make_contact_sheet(frame_paths, contact_sheet, f"Go2 路线关键帧 · {args.scene}")
        final_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        final_quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        displacement = float(np.linalg.norm(final_position[:2] - initial_position[:2]))
        z_min = min(
            json.loads(line)["base_position_world"][2]
            for line in trajectory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        ground_clearance_min = (
            float(z_min - selected_ground_z) if selected_ground_z is not None else None
        )
        ground_clearance_final = (
            float(final_position[2] - selected_ground_z) if selected_ground_z is not None else None
        )
        frame_records = [json.loads(line) for line in frame_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        timestamps = [float(record["timestamp_s"]) for record in frame_records]
        if args.lightweight_three_panel:
            centre_name = next(cal["id"] for cal in camera_calibrations if cal["role"] == "center_forward_pinhole")
            modality_complete = bool(frame_records) and all(
                set(record["modalities"]) == {centre_name, "overview", "review"}
                for record in frame_records
            )
            all_rgb_nonempty = bool(frame_records) and all(
                bool(record["modalities"]["overview"]["nonempty_rgb"])
                for record in frame_records
            )
            all_depth_present = None
            onboard_depth_usable = bool(frame_records) and all(
                float(record["modalities"][centre_name]["finite_depth_ratio"]) > 0.01
                for record in frame_records
            )
        else:
            modality_complete = bool(frame_records) and all(
                set(record["modalities"]) == {cal["id"] for cal in camera_calibrations} | {"overview"}
                for record in frame_records
            )
            all_rgb_nonempty = bool(frame_records) and all(
                modality["rgb_metrics"]["nonempty_rgb"]
                for record in frame_records
                for modality in record["modalities"].values()
            )
            all_depth_present = bool(frame_records) and all(
                modality["rgb_metrics"].get("finite_depth_ratio") is not None
                for record in frame_records
                for modality in record["modalities"].values()
            )
            onboard_depth_usable = bool(frame_records) and all(
                float(record["modalities"][calibration["id"]]["rgb_metrics"].get("finite_depth_ratio") or 0.0) > 0.01
                for record in frame_records
                for calibration in camera_calibrations
            )
        phase1_report = None
        if phase1_vehicle_actor is not None:
            phase1_report = {
                "stage": "phase_one_stop_yield_resume",
                "avoidance_mode": args.phase1_avoidance_mode,
                "vehicle": phase1_vehicle_actor.summary(),
                "go2_avoidance": phase1_yield_controller.summary(),
                "responsibility_contract": {
                    "vehicle_observes_go2": False,
                    "vehicle_schedule_input": "simulation timestamp only",
                    "go2_observes_vehicle": True,
                    "go2_base_pose_prescribed": False,
                    "locomotion_execution": "frozen policy joint actions through PhysX",
                },
            }
            write_json(metadata / "phase1_dynamic_vehicle_summary.json", phase1_report)
        traffic_summary = None
        if traffic_manager is not None:
            # The final geometry readback must observe the same post-env.step
            # proxy transforms used by the last RGB/depth frame.
            traffic_manager.refresh_visual_transforms()
            traffic_summary = traffic_manager.summary()
            write_json(metadata / "scene10_continuous_traffic_summary.json", traffic_summary)
            write_json(metadata / "traffic_states.json", traffic_manager.states)
        people_summary = None
        if people_manager is not None:
            people_manager.refresh_visual_transforms()
            people_summary = people_manager.summary()
            write_json(metadata / "animated_pedestrian_summary.json", people_summary)
        traffic_visual_alignment = (
            traffic_summary.get("visual_geometry_alignment", [])
            if traffic_summary is not None
            else []
        )
        # A continuous-traffic vehicle that has completed a route is deliberately
        # deactivated and parked off-map during its random cooldown.  It has no
        # rendered bbox at that instant and must not invalidate alignment evidence
        # for the vehicles that are actually visible in the captured frame.
        traffic_active_visual_alignment = [
            row
            for row in traffic_visual_alignment
            if row.get("expected_location") == "route"
        ]
        traffic_visual_centers_valid = bool(traffic_active_visual_alignment) and all(
            bool(row.get("bbox_valid"))
            and row.get("bbox_to_expected_visual_center_xy_error_m") is not None
            and float(row["bbox_to_expected_visual_center_xy_error_m"])
            <= args.maximum_traffic_visual_center_error
            for row in traffic_active_visual_alignment
        )
        traffic_fast_proxy_heading_valid = bool(
            traffic_visual_centers_valid
            and traffic_summary is not None
            and all(
                row.get("runtime_proxy_front_heading_deg") is not None
                for row in traffic_active_visual_alignment
            )
            and float(traffic_summary["maximum_heading_motion_error_deg"]) <= 0.5
        )
        checks = {
            "actual_articulated_go2_loaded": bool(robot.is_initialized),
            "existing_controller_loaded_without_training": True,
            "execution_mode": args.execution_mode,
            "trajectory_carrier_proxy_used": trajectory_carrier is not None,
            "trajectory_carrier_pose_followed": (
                maximum_carrier_xy_error < 1.0e-3 and maximum_carrier_yaw_error < 1.0e-3
                if trajectory_carrier is not None
                else None
            ),
            "contact_and_gait_evidence_valid": trajectory_carrier is None,
            "complete_step_trajectory_saved": bool(
                step_times
                and (
                    len(step_times) == total_steps
                    or route_stuck
                    or route_collision_abort
                    or post_goal_stop
                )
            ),
            "timestamp_and_frame_index_strictly_increasing": all(
                later > earlier for earlier, later in zip(timestamps, timestamps[1:])
            ),
            "all_requested_modalities_indexed": modality_complete,
            "all_rgb_nonempty": all_rgb_nonempty,
            "all_depth_arrays_recorded": all_depth_present,
            "lightweight_depth_rendered_for_every_review_frame": (
                onboard_depth_usable if args.lightweight_three_panel else None
            ),
            "onboard_depth_has_finite_scene_geometry": onboard_depth_usable,
            "coda_delivery_complete": (
                coda_lidar_scan_count == frame_count and frame_count > 0
                if coda_writer is not None
                else None
            ),
            "coda_camera_lidar_sync_within_10ms": (
                coda_max_sync_offset_ns is not None and coda_max_sync_offset_ns <= 10_000_000
                if coda_writer is not None
                else None
            ),
            "coda_delivery_schema_valid": (
                coda_validation is not None and coda_validation.get("status") == "success"
                if coda_writer is not None
                else None
            ),
            "coda_three_pinhole_rig_valid": (
                coda_validation is not None
                and bool(coda_validation.get("three_pinhole_required"))
                and coda_validation.get("status") == "success"
                if args.coda_pinhole_rig
                else None
            ),
            "robot_horizontal_motion_observed": maximum_distance_from_start > 0.20,
            "closed_loop_waypoint_control_used": (
                route_mode in ("waypoint_loop", "global_planned") if trajectory_carrier is None else False
            ),
            "route_goal_reached": (
                None
                if trajectory_carrier is not None
                else bool(
                    waypoint_follower.route_complete
                    if waypoint_follower.stop_at_final_waypoint or safe_shuttle_plan is not None
                    else True
                )
            ),
            "route_completed_without_stuck_detection": not route_stuck,
            "route_completed_without_sustained_collision": not route_collision_abort,
            "route_completed_without_non_foot_collision": (
                collision_steps == 0 if trajectory_carrier is None else None
            ),
            "isolated_non_foot_contacts_only": (
                collision_steps > 0 and not route_collision_abort
                if trajectory_carrier is None
                else None
            ),
            "road_corridor_prevalidated": (
                bool(road_route_probe is None or road_route_probe.get("accepted"))
                if route_mode != "global_planned"
                else None
            ),
            "global_route_prevalidated": bool(global_route_plan is not None and global_route_plan.get("accepted")),
            "global_route_plan_record": (
                {
                    "path_length_m": (global_route_plan.get("astar") or {}).get("path_length_m"),
                    "target_route_length_m": global_route_plan.get("target_route_length_m"),
                    "length_band_used_m": global_route_plan.get("length_band_used_m"),
                    "max_curvature_rad_per_m": (global_route_plan.get("spline") or {}).get(
                        "max_curvature_rad_per_m"
                    ),
                    "total_turn_rad": (global_route_plan.get("turn") or {}).get("total_turn_rad"),
                    "total_turn_deg": (global_route_plan.get("turn") or {}).get("total_turn_deg"),
                    "min_total_turn_rad": (global_route_plan.get("turn") or {}).get("min_total_turn_rad"),
                    "validation_segment_count": (global_route_plan.get("validation") or {}).get("segment_count"),
                    "failed_segment_indices": (global_route_plan.get("validation") or {}).get(
                        "failed_segment_indices"
                    ),
                    "cache_hit": (global_route_plan.get("grid") or {}).get("cache_hit"),
                }
                if global_route_plan is not None
                else None
            ),
            "right_angle_route_prevalidated": (
                bool(right_angle_route_probe.get("accepted"))
                if right_angle_route_probe is not None
                else None
            ),
            "right_angle_route_completed": (
                bool(waypoint_follower.route_complete)
                if right_angle_route_probe is not None
                else None
            ),
            "safe_shuttle_completed_cycle": (
                waypoint_follower.loop_count >= 1 if safe_shuttle_plan is not None else None
            ),
            "route_ground_collider_found": selected_ground_z is not None,
            "robot_supported_above_route_ground": bool(
                ground_clearance_min is not None
                and ground_clearance_final is not None
                and ground_clearance_min > 0.05
                and ground_clearance_final > 0.15
            ),
            "all_root_poses_finite": nonfinite_steps == 0,
            "video_readback_verified": video_metadata["frame_count"] == frame_count,
            "locomotion_training_performed": False,
            "strict_document_resolution_exact": (
                [args.sensor_width, args.sensor_height] == camera_calibrations[0]["image_size"]
                if args.strict_document_calibration
                else None
            ),
            "strict_document_resolution_policy_satisfied": (
                (
                    [args.sensor_width, args.sensor_height] == camera_calibrations[0]["image_size"]
                    or (
                        args.allow_scaled_document_calibration
                        and math.isclose(
                            float(args.sensor_width) / float(args.sensor_height),
                            float(camera_calibrations[0]["image_size"][0])
                            / float(camera_calibrations[0]["image_size"][1]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    )
                )
                if args.strict_document_calibration
                else None
            ),
            "strict_document_intrinsics_scaled": (
                [args.sensor_width, args.sensor_height] != camera_calibrations[0]["image_size"]
                if args.strict_document_calibration
                else None
            ),
            "strict_document_k_d_mapping_applied": (
                len(strict_mapping_metadata) == len(camera_calibrations)
                if args.strict_document_calibration
                else None
            ),
            "strict_inverse_mapping_numerically_converged": (
                all(
                    row.get("max_inverse_residual_normalized") is not None
                    and float(row["max_inverse_residual_normalized"]) < 1e-8
                    for row in strict_mapping_metadata.values()
                )
                if args.strict_document_calibration
                else None
            ),
            "perception_debug_visualization_disabled": not bool(cfg.commands.base_velocity.debug_vis),
            "phase1_dynamic_vehicle_enabled": phase1_vehicle_actor is not None,
            "phase1_original_vehicle_asset_loaded": (
                phase1_vehicle_actor.evidence["mesh_count"] >= 1
                and phase1_vehicle_actor.evidence["collision_api_count"] >= 1
                if phase1_vehicle_actor is not None
                else None
            ),
            "phase1_vehicle_schedule_independent_of_go2": (
                True if phase1_vehicle_actor is not None else None
            ),
            "phase1_vehicle_completed_crossing": (
                bool(phase1_vehicle_state.get("completed"))
                if phase1_vehicle_state is not None
                else None
            ),
            "phase1_go2_avoidance_intervention_observed": (
                phase1_yield_controller.intervention_steps > 0
                if phase1_vehicle_actor is not None and args.phase1_avoidance_mode == "yield"
                else None
            ),
            "phase1_go2_full_stop_observed": (
                phase1_yield_controller.full_stop_steps > 0
                if phase1_vehicle_actor is not None and args.phase1_avoidance_mode == "yield"
                else None
            ),
            "phase1_positive_analytic_clearance": (
                phase1_yield_controller.minimum_clearance > 0.0
                if phase1_vehicle_actor is not None and args.phase1_avoidance_mode == "yield"
                else None
            ),
            "scene10_continuous_traffic_enabled": traffic_manager is not None,
            "scene10_continuous_traffic_all_vehicles_spawned": (
                min(traffic_summary["spawn_counts"]) >= 1
                if traffic_summary is not None
                else None
            ),
            "scene10_continuous_traffic_dynamic_overlap_free": (
                traffic_summary["dynamic_obb_overlap_events"] == 0
                if traffic_summary is not None
                else None
            ),
            "scene10_continuous_traffic_static_overlap_free": (
                traffic_summary["static_obb_overlap_events"] == 0
                if traffic_summary is not None
                else None
            ),
            "scene10_continuous_traffic_go2_overlap_free": (
                traffic_summary["go2_vehicle_overlap_events"] == 0
                if traffic_summary is not None
                else None
            ),
            "scene10_continuous_traffic_go2_clearance_satisfied": (
                traffic_summary["minimum_go2_vehicle_clearance_m"] is not None
                and float(traffic_summary["minimum_go2_vehicle_clearance_m"])
                >= args.minimum_go2_vehicle_clearance
                if traffic_summary is not None
                else None
            ),
            "scene10_continuous_traffic_visual_centers_valid": (
                traffic_visual_centers_valid if traffic_summary is not None else None
            ),
            "scene10_continuous_traffic_heading_motion_valid": (
                float(traffic_summary["maximum_heading_motion_error_deg"]) <= 3.0
                if traffic_summary is not None
                else None
            ),
            "scene10_continuous_traffic_rendered_heading_valid": (
                (
                    traffic_summary["maximum_rendered_body_axis_error_deg"] is not None
                    and float(traffic_summary["maximum_rendered_body_axis_error_deg"])
                    <= args.maximum_rendered_body_axis_error
                )
                or traffic_fast_proxy_heading_valid
                if traffic_summary is not None
                else None
            ),
            "scene10_continuous_traffic_visible_near_go2": (
                float(traffic_summary["third_person_dynamic_pass_max_duration_s"])
                >= args.minimum_visible_vehicle_pass_duration
                if traffic_summary is not None
                else None
            ),
        }
        required_execution_checks = [
                "actual_articulated_go2_loaded",
                "existing_controller_loaded_without_training",
                "complete_step_trajectory_saved",
                "timestamp_and_frame_index_strictly_increasing",
                "all_requested_modalities_indexed",
                "all_rgb_nonempty",
                "onboard_depth_has_finite_scene_geometry",
                "robot_horizontal_motion_observed",
                "route_completed_without_stuck_detection",
                "route_completed_without_sustained_collision",
                "route_ground_collider_found",
                "robot_supported_above_route_ground",
                "all_root_poses_finite",
                "video_readback_verified",
                "perception_debug_visualization_disabled",
        ]
        required_execution_checks.append(
            "lightweight_depth_rendered_for_every_review_frame"
            if args.lightweight_three_panel
            else "all_depth_arrays_recorded"
        )
        if coda_writer is not None:
            required_execution_checks.extend(
                [
                    "coda_delivery_complete",
                    "coda_camera_lidar_sync_within_10ms",
                    "coda_delivery_schema_valid",
                ]
            )
            if args.coda_pinhole_rig:
                required_execution_checks.append("coda_three_pinhole_rig_valid")
        if trajectory_carrier is None:
            required_execution_checks.extend(
                [
                    "closed_loop_waypoint_control_used",
                    "route_goal_reached",
                ]
            )
        else:
            required_execution_checks.append("trajectory_carrier_pose_followed")
        if route_mode == "global_planned":
            required_execution_checks.append("global_route_prevalidated")
        else:
            required_execution_checks.append("road_corridor_prevalidated")
        if phase1_vehicle_actor is not None and args.phase1_avoidance_mode == "yield":
            required_execution_checks.extend(
                [
                    "phase1_original_vehicle_asset_loaded",
                    "phase1_vehicle_schedule_independent_of_go2",
                    "phase1_vehicle_completed_crossing",
                    "phase1_go2_avoidance_intervention_observed",
                    "phase1_go2_full_stop_observed",
                    "phase1_positive_analytic_clearance",
                ]
            )
        if traffic_manager is not None:
            required_execution_checks.extend(
                [
                    "scene10_continuous_traffic_enabled",
                    "scene10_continuous_traffic_all_vehicles_spawned",
                    "scene10_continuous_traffic_dynamic_overlap_free",
                    "scene10_continuous_traffic_static_overlap_free",
                    "scene10_continuous_traffic_go2_overlap_free",
                    "scene10_continuous_traffic_go2_clearance_satisfied",
                    "scene10_continuous_traffic_visual_centers_valid",
                    "scene10_continuous_traffic_heading_motion_valid",
                    "scene10_continuous_traffic_rendered_heading_valid",
                    "scene10_continuous_traffic_visible_near_go2",
                ]
            )
        execution_ok = all(bool(checks[key]) for key in required_execution_checks)
        navigation_ok = (
            bool(
                checks.get("route_goal_reached")
                and checks.get("route_completed_without_stuck_detection")
                and checks.get("route_completed_without_sustained_collision")
                and checks.get("all_root_poses_finite")
            )
            if trajectory_carrier is None
            else bool(checks.get("trajectory_carrier_pose_followed"))
        )
        data_integrity_keys = [
            "complete_step_trajectory_saved",
            "timestamp_and_frame_index_strictly_increasing",
            "all_requested_modalities_indexed",
            "all_rgb_nonempty",
            "onboard_depth_has_finite_scene_geometry",
            "video_readback_verified",
        ]
        data_integrity_keys.append(
            "lightweight_depth_rendered_for_every_review_frame"
            if args.lightweight_three_panel
            else "all_depth_arrays_recorded"
        )
        data_integrity_ok = all(bool(checks[key]) for key in data_integrity_keys)
        if args.strict_document_calibration:
            execution_ok = execution_ok and all(
                bool(checks[key])
                for key in (
                    "strict_document_resolution_policy_satisfied",
                    "strict_document_k_d_mapping_applied",
                    "strict_inverse_mapping_numerically_converged",
                )
            )
        step_ms = np.asarray(step_times, dtype=np.float64) * 1000.0
        environment = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": platform.node(),
            "gpu_index": args.gpu,
            "gpu": gpu_snapshot(args.gpu),
            "driver": gpu_snapshot(args.gpu).get("driver"),
            "isaac_sim_version": package_version("isaacsim"),
            "replicator_version": package_version("isaacsim-replicator"),
            "isaac_lab_release": (PROJECT_ROOT / "repos" / "IsaacLab" / "VERSION").read_text(encoding="utf-8").strip(),
            "isaac_lab_commit": run_text(["git", "-C", str(PROJECT_ROOT / "repos" / "IsaacLab"), "rev-parse", "HEAD"]),
            "torch_version": package_version("torch"),
            "python_version": sys.version.replace("\n", " "),
            "kit_version": appearance.json_default(omni.kit.app.get_app().get_kit_version()),
            "git_commit": run_text(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]),
            "command_line": [sys.executable, *sys.argv],
            "renderer": "RayTracedLighting",
            "kit_experience": str(PROJECT_ROOT / "configs" / "isaac45_urbanverse_go2_headless.kit"),
            "sensor_resolution": [args.sensor_width, args.sensor_height],
            "overview_resolution": [args.overview_width, args.overview_height],
            "capture_fps_requested": args.capture_fps,
            "capture_fps_actual": actual_capture_fps,
            "review_playback_speed": args.playback_speed,
            "review_video_fps": actual_capture_fps * args.playback_speed,
            "dynamic_agent_motion_blur": {
                "requested": bool(args.dynamic_agent_motion_blur),
                "rtx_motion_blur_enabled": bool(settings.get("/rtx/post/motionblur/enabled")),
                "rtx_motion_blur_camelcase_enabled": bool(settings.get("/rtx/post/motionBlur/enabled")),
            },
            "lightweight_three_panel": bool(args.lightweight_three_panel),
            "source_usd": str(source_usd),
            "source_usd_sha256": sha256(source_usd),
            "source_tar": str(source_tar) if source_tar else None,
            "source_tar_sha256": sha256(source_tar),
            "wrapper_path": str(wrapper_path),
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "random_seed": args.seed,
            "appearance_profile": args.appearance_profile,
            "appearance_application": appearance_application,
            "collection_light_scale": light_scale_overrides,
            "controller": {
                "role": (
                    "joint-animation source only; base pose is prescribed by the navigation trajectory carrier"
                    if trajectory_carrier is not None
                    else "existing low-level trajectory executor; not trained or tuned by this project"
                ),
                "profile": args.locomotion_profile,
                "policy_kind": args.policy_kind,
                "external_policy": external_policy_metadata,
                "official_task": official_task,
                "official_checkpoint_url": controller_profile["checkpoint_url"] if args.policy_kind == "torchscript" else None,
                "terrain_observation": controller_profile["terrain_observation"],
                "rough_height_scan_proxy": height_scan_proxy,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "policy_path": str(policy_path),
                "policy_sha256": sha256(policy_path),
                "pre_capture_warmup": controller_warmup,
            },
            "source_stage_metadata": source_stage_metadata,
            "route_anchor": anchor,
            "ground_probe": ground_probe,
            "road_route_probe": road_route_probe,
            "safe_shuttle_plan": safe_shuttle_plan,
            "source_camera": source_camera_meta,
            "capture_contract": (
                "Lightweight success test: physics-rate trajectory/contact/control plus centre depth rendered into review frames and third-person RGB; no onboard RGB or float32 depth arrays."
                if args.lightweight_three_panel
                else "Complete raw run; no model sample schema, history length, prediction horizon or waypoint count is fixed."
            ),
            "strict_document_calibration": bool(args.strict_document_calibration),
            "coda_three_pinhole_rig": bool(args.coda_pinhole_rig),
            "coda_pinhole_horizontal_fov_deg": (
                float(args.coda_pinhole_horizontal_fov_deg) if args.coda_pinhole_rig else None
            ),
            "coda_pinhole_side_yaw_deg": (
                float(args.coda_pinhole_side_yaw_deg) if args.coda_pinhole_rig else None
            ),
            "allow_scaled_document_calibration": bool(args.allow_scaled_document_calibration),
            "document_original_resolution": camera_calibrations[0]["image_size"],
            "render_resolution": [args.sensor_width, args.sensor_height],
            "document_intrinsics_scale_xy": [
                float(args.sensor_width) / float(camera_calibrations[0]["image_size"][0]),
                float(args.sensor_height) / float(camera_calibrations[0]["image_size"][1]),
            ],
            "strict_mapping": strict_mapping_metadata if args.strict_document_calibration else None,
            "requested_route_duration_s": args.route_duration_seconds,
            "actual_control_schedule_duration_s": total_duration,
            "navigation": {
                "mode": route_mode,
                "execution_mode": args.execution_mode,
                "scene_config": navigation_config,
                "waypoints_world_xy": waypoints_world,
                "feedback_source": (
                    "prescribed timestamp-indexed carrier pose on the admitted route"
                    if trajectory_carrier is not None
                    else "live Go2 base XY and yaw at every physics step"
                ),
                "pre_capture_controller_warmup": controller_warmup,
                "route_admission": "PhysX road-corridor probe" if road_route_probe is not None else "configured waypoints",
                "stop_at_final_waypoint": (
                    waypoint_follower.stop_at_final_waypoint if waypoint_follower is not None else False
                ),
                "safe_shuttle": safe_shuttle_plan is not None,
                "trajectory_carrier": (
                    {
                        "speed": trajectory_carrier.speed,
                        "travel_distance": trajectory_carrier.distance,
                        "travel_duration_s": trajectory_carrier.travel_duration_s,
                        "cycle_duration_s": trajectory_carrier.cycle_duration_s,
                        "contact_and_gait_evidence_valid": False,
                    }
                    if trajectory_carrier is not None
                    else None
                ),
            },
            "phase1_dynamic_vehicle": phase1_report,
            "scene10_continuous_traffic": {
                "enabled": traffic_manager is not None,
                "config": (
                    str(metadata / "scene10_continuous_traffic_config.json")
                    if traffic_manager is not None
                    else None
                ),
                "summary": traffic_summary,
                "preauthored_sources": preauthored_traffic_sources,
            },
            "animated_pedestrians": {
                "enabled": people_manager is not None,
                "summary": people_summary,
                "preauthored_people": preauthored_people,
            },
            "motion_vectors_requested": not args.omit_motion_vectors,
            "derived_artifact_sampling": {
                "depth_visualization_stride_frames": args.derived_visualization_stride,
                "full_resolution_metrics_stride_frames": args.full_metrics_stride,
                "raw_rgb_every_capture": True,
                "raw_float32_depth_every_capture": True,
                "trajectory_control_pose_contact_every_physics_step": True,
            },
        }
        camera_metadata = {
            "rig_contract": (
                {
                    "model": "ideal pinhole",
                    "camera_names": ["cam_front", "cam_front_left", "cam_front_right"],
                    "horizontal_fov_deg": float(args.coda_pinhole_horizontal_fov_deg),
                    "side_yaw_deg_body_convention": float(args.coda_pinhole_side_yaw_deg),
                    "body_frame_convention": "+X forward, +Y left, +Z up",
                    "distortion": [0.0, 0.0, 0.0, 0.0],
                    "simultaneous_render_timestamps": True,
                    "shared_render_and_appearance_settings": True,
                }
                if args.coda_pinhole_rig
                else None
            ),
            "mount_policy": "The three navigation sensors are Isaac Lab CameraCfg children of the articulated Go2 base, created before Fabric initialization. The overview camera is a separate reproducible chase view and is not a training sensor.",
            "orientation_policy": (
                "The centre CameraExt is applied in ROS optical convention. The internally inconsistent side RPY blocks are resolved as horizontal outward-facing vehicle/body-yaw mounts."
                if args.strict_document_calibration
                else "Onboard cameras follow the full articulated base attitude in Isaac Lab world convention (+X forward, +Z up), internally converted to OpenGL. No image rotation or roll/pitch stabilization is applied."
            ),
            "creation_stage": "before gym.make and Fabric initialization",
            "resolved_prim_paths": {row["name"]: str(row["camera_prim"].GetPath()) for row in camera_rows},
            "mount_local_transforms": {
                row["name"]: row.get("mount_local_transform") for row in camera_rows if row["name"] != "overview"
            },
            "overview_chase_view": {
                "distance_behind_m": args.overview_chase_distance,
                "lateral_offset_m": args.overview_chase_lateral_offset,
                "height_above_base_m": args.overview_chase_height,
                "target_forward_m": args.overview_target_forward,
                "target_height_above_base_m": args.overview_target_height,
                "focal_length_mm": args.overview_focal_length,
            },
            "calibrations": camera_calibrations,
            "render_resolution": [args.sensor_width, args.sensor_height],
            "document_original_resolution": camera_calibrations[0]["image_size"],
            "document_intrinsics_scale_xy": [
                float(args.sensor_width) / float(camera_calibrations[0]["image_size"][0]),
                float(args.sensor_height) / float(camera_calibrations[0]["image_size"][1]),
            ],
            "requested_capture_fps": float(args.capture_fps),
            "actual_capture_fps": actual_capture_fps,
            "sensor_update_period_s": {
                name: float(definition["update_period_s"])
                for name, definition in camera_definitions.items()
            },
            "derived_artifact_sampling": {
                "depth_visualization_stride_frames": args.derived_visualization_stride,
                "full_resolution_metrics_stride_frames": args.full_metrics_stride,
                "raw_rgb_and_float32_depth_stride_frames": (
                    None if args.lightweight_three_panel else 1
                ),
            },
            "physics_pose_control_rate_note": (
                "Trajectory, control, action, pose and contact remain recorded at every physics step; only third-person RGB and the composed depth review frame are saved at capture_fps."
                if args.lightweight_three_panel
                else "Trajectory, control, action, pose and contact remain recorded at every physics step; RGB/depth sensors refresh and save at capture_fps."
            ),
            "frame_index": str(frame_index_path),
            "appearance_application": appearance_application,
            "strict_document_calibration": bool(args.strict_document_calibration),
            "allow_scaled_document_calibration": bool(args.allow_scaled_document_calibration),
            "strict_mapping": strict_mapping_metadata if args.strict_document_calibration else None,
        }
        write_json(metadata / "environment.json", environment)
        write_json(metadata / "camera.json", camera_metadata)
        metrics = {
            "requested_step_count": total_steps,
            "step_count": len(step_times),
            "frame_count": frame_count,
            "step_dt_s": dt,
            "capture_fps": actual_capture_fps,
            "requested_simulation_duration_s": total_duration,
            "simulation_duration_s": len(step_times) * dt,
            "wall_time": {
                "initialization_before_capture_s": capture_loop_started - started,
                "capture_loop_s": capture_loop_wall_s,
                "video_encoding_s": video_encoding_wall_s,
                "end_to_end_s": time.perf_counter() - started,
            },
            "step_p50_ms": float(np.percentile(step_ms, 50)),
            "step_p95_ms": float(np.percentile(step_ms, 95)),
            "initial_base_position": initial_position.tolist(),
            "final_base_position": final_position.tolist(),
            "final_base_quaternion_wxyz": final_quat.tolist(),
            "horizontal_displacement_stage_units": displacement,
            "maximum_distance_from_start_stage_units": maximum_distance_from_start,
            "minimum_base_z_stage_units": float(z_min),
            "estimated_ground_z_stage_units": anchor["estimated_ground_z"],
            "physx_probed_ground_z_stage_units": selected_ground_z,
            "minimum_ground_clearance_stage_units": ground_clearance_min,
            "final_ground_clearance_stage_units": ground_clearance_final,
            "body_collision_step_count": collision_steps,
            "contact_evidence_valid": trajectory_carrier is None,
            "foot_support_step_count": foot_support_steps,
            "route_stuck": route_stuck,
            "stuck_event": stuck_event,
            "route_collision_abort": route_collision_abort,
            "collision_event": collision_event,
            "route_goal_reached_at_s": post_goal_state.get("goal_reached_at_s"),
            "post_goal_hold_s": post_goal_state.get("held_s"),
            "post_goal_capture_stop": post_goal_stop,
            "waypoint_loop_count": (
                trajectory_carrier.loop_count if trajectory_carrier is not None else waypoint_follower.loop_count
            ),
            "route_goal_reached": (
                None
                if trajectory_carrier is not None
                else waypoint_follower.route_complete
                if waypoint_follower.stop_at_final_waypoint or safe_shuttle_plan is not None
                else None
            ),
            "maximum_carrier_xy_error_stage_units": (
                maximum_carrier_xy_error if trajectory_carrier is not None else None
            ),
            "maximum_carrier_yaw_error_rad": (
                maximum_carrier_yaw_error if trajectory_carrier is not None else None
            ),
            "nonfinite_pose_step_count": nonfinite_steps,
            "phase1_dynamic_vehicle": phase1_report,
            "scene10_continuous_traffic": traffic_summary,
            "video": video_metadata,
        }
        summary.update(
            {
                "status": "success" if execution_ok else "failed",
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_s": round(time.perf_counter() - started, 3),
                "environment": environment,
                "metrics": metrics,
                "checks": checks,
                "failure_stage": None if execution_ok else "route_capture_validation",
                "error": None if execution_ok else "one or more raw-run/Go2 motion validation checks failed",
                "status_components": {
                    "navigation_task": "success" if navigation_ok else "failed",
                    "data_capture": "success" if data_integrity_ok else "failed",
                    "traffic_environment": (
                        "success"
                        if traffic_manager is not None
                        and all(
                            bool(checks[key])
                            for key in (
                                "scene10_continuous_traffic_all_vehicles_spawned",
                                "scene10_continuous_traffic_dynamic_overlap_free",
                                "scene10_continuous_traffic_static_overlap_free",
                                "scene10_continuous_traffic_go2_overlap_free",
                                "scene10_continuous_traffic_go2_clearance_satisfied",
                                "scene10_continuous_traffic_visual_centers_valid",
                                "scene10_continuous_traffic_heading_motion_valid",
                                "scene10_continuous_traffic_rendered_heading_valid",
                                "scene10_continuous_traffic_visible_near_go2",
                            )
                        )
                        else "failed"
                        if traffic_manager is not None
                        else "not_requested"
                    ),
                    "isolated_non_foot_contacts": (
                        "warning" if collision_steps > 0 and not route_collision_abort else "none"
                    ),
                },
                "warnings": (
                    [
                        f"Observed {collision_steps} isolated non-foot contact step(s); "
                        "no sustained collision abort was triggered."
                    ]
                    if collision_steps > 0 and not route_collision_abort
                    else []
                ),
                "assessment": (
                    "Actual articulated Go2 and camera rig on a timestamp-prescribed, PhysX-prevalidated road trajectory. This validates synchronized navigation-data capture; gait, contacts and obstacle traversal are explicitly not validated."
                    if trajectory_carrier is not None
                    else "Actual articulated Go2 with the selected frozen low-level controller and live-pose waypoint feedback; this evaluates route-data capture, not gait learning."
                ),
                "limitations": [
                    (
                        "Strict mode explicitly remaps native RGB/depth into the recorded OpenCV K/D pixel domain; "
                        "when scaled resolution is allowed, fx/fy/cx/cy are scaled while D and CameraExt remain unchanged."
                        if args.strict_document_calibration
                        else "The three CoDa cameras are native ideal pinhole renders with output-resolution K, zero distortion and no post-render image remapping."
                        if args.coda_pinhole_rig
                        else "The supplied OpenCV fisheye D values are retained but Isaac 4.5 native fisheyePolynomial is not claimed equivalent."
                    ),
                    *(
                        []
                        if args.coda_pinhole_rig
                        else [
                            "The centre block says model=fisheye while its ID and project text say pinhole; strict mode records the explicit pinhole/radtan interpretation.",
                            "The side-camera ROS-optical RPY is mathematically inconsistent with a horizontal side view; this run uses the recorded horizontal-outward interpretation and does not claim final real-camera extrinsic equivalence.",
                        ]
                    ),
                    "Source metersPerUnit=0.01 conflicts with the visually validated metre-like route coordinates; all raw values are retained without silently relabelling units.",
                    (
                        "Five continuous kinematic vehicles provide simulator ground-truth traffic; Go2 follows a spatially separated NearRoad route and does not perceive or avoid those vehicles in this stage."
                        if traffic_manager is not None
                        else "Phase one uses one timestamp-scheduled original SUV and simulator ground-truth state for Go2 yielding; perception-derived tracking, lateral detours and multi-agent traffic are not yet evaluated."
                        if phase1_vehicle_actor is not None
                        else "Real traffic and pedestrian dynamics are outside this ego-motion run."
                    ),
                    (
                        "The base pose is prescribed by a navigation trajectory carrier. The frozen policy only animates joints, so collision/contact/foot-support values are diagnostic and cannot be used as physics or locomotion evidence."
                        if trajectory_carrier is not None
                        else "Base motion is generated through the frozen locomotion policy and PhysX."
                    ),
                    (
                        "The rough policy is a low-level executor for limited terrain variation, not a scene-level obstacle planner and not proof that arbitrary curbs or stairs are traversable."
                        if args.locomotion_profile == "rough"
                        else "The flat policy requires a prevalidated traversable route; it is not an obstacle planner and is not evidence that curbs or stairs are traversable."
                    ),
                    (
                        "The safe shuttle turns at admitted endpoints and returns forward through the same corridor; "
                        "this validates controlled route capture, not autonomous obstacle avoidance or route planning."
                        if safe_shuttle_plan is not None
                        else "No prevalidated road-shuttle behavior was requested."
                    ),
                ],
                "evidence": {
                    "trajectory": str(trajectory_path),
                    "frame_index": str(frame_index_path),
                    "reference_route": str(metadata / "reference_route.json"),
                    "control_schedule": str(metadata / "control_schedule.json"),
                    "controller_warmup": str(controller_warmup_path),
                    "road_route_probe": (
                        str(metadata / "road_route_probe.json") if road_route_probe is not None else None
                    ),
                    "safe_shuttle_plan": (
                        str(metadata / "safe_shuttle_plan.json") if safe_shuttle_plan is not None else None
                    ),
                    "phase1_dynamic_vehicle": (
                        str(metadata / "phase1_dynamic_vehicle_summary.json")
                        if phase1_vehicle_actor is not None
                        else None
                    ),
                    "scene10_continuous_traffic_config": (
                        str(metadata / "scene10_continuous_traffic_config.json")
                        if traffic_manager is not None
                        else None
                    ),
                    "scene10_continuous_traffic_summary": (
                        str(metadata / "scene10_continuous_traffic_summary.json")
                        if traffic_manager is not None
                        else None
                    ),
                    "traffic_states": (
                        str(metadata / "traffic_states.json")
                        if traffic_manager is not None
                        else None
                    ),
                    "video": str(video_path),
                    "contact_sheet": str(contact_sheet),
                },
            }
        )
        write_json(metadata / "metrics.json", metrics)
        write_json(summary_path, summary)
        result_code = 0 if execution_ok else 1
        safe_print(f"RESULT summary={summary_path} status={summary['status']}", flush=True)
    except Exception as exc:
        formatted_traceback = traceback.format_exc()
        safe_print(formatted_traceback, file=sys.stderr, flush=True)
        summary.update(
            {
                "status": "failed",
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_s": round(time.perf_counter() - started, 3),
                "failure_stage": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": formatted_traceback,
            }
        )
        write_json(summary_path, summary)
        result_code = 1
    finally:
        if coda_lidar_annotator is not None:
            try:
                coda_lidar_annotator.detach()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
