#!/usr/bin/env python3
"""Render a small straight-line traffic flow in CraftBench scene_10.

The first traffic milestone intentionally excludes Go2 and vehicle dynamics.
It uses qualified native vehicle geometry, straight kinematic trajectories,
smooth wrap-around outside the main camera focus, and aligned box proxies.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


QUALIFIED_ASSET_ID = "d90c7f830f9c41398bb55de4a2e001be"
BASE_QUAT_XYZW = np.asarray([0.5000000008902851, -0.49999999910971493, -0.49999999703181697, 0.500000002968183])
# The local-bound center already captures the source GLB's off-center pivot.
# Extra empirical corrections shift the visible mesh away from its collision box.
ROOT_CORRECTION_XYZ = np.zeros(3, dtype=np.float64)
VEHICLE_SIZE_XYZ = np.asarray([3.51364523228267, 1.7577024708833733, 1.8499994277954108])
VEHICLE_HALF_HEIGHT = float(VEHICLE_SIZE_XYZ[2] / 2.0)
TRAFFIC_LANE_Y = 514.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--vehicle-asset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--ext-folder", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=12.0)
    return parser.parse_args()


def now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_stage(context, app, max_updates: int = 1200) -> None:
    for _ in range(max_updates):
        app.update()
        try:
            status = tuple(context.get_stage_loading_status())
        except Exception:
            status = ()
        if not status or (len(status) >= 3 and int(status[-1]) == 0):
            return
    raise RuntimeError("stage loading timed out")


def multiply_quaternions_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def rotate_vector_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[:3]
    scalar = quat[3]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + scalar * vector)


def rgb_array(annotator) -> np.ndarray:
    payload = annotator.get_data()
    if isinstance(payload, dict):
        payload = payload.get("data")
    array = np.asarray(payload)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"unexpected RGB annotator shape: {array.shape}")
    return np.ascontiguousarray(array[:, :, :3].astype(np.uint8, copy=False))


def annotate(frame: np.ndarray, timestamp_s: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((12, 12, 305, 55), radius=7, fill=(8, 12, 18, 165))
    draw.text((22, 18), "scene_10 · 2 dynamic SUVs + static traffic", fill=(255, 255, 255, 255))
    draw.text((22, 36), f"t={timestamp_s:05.2f}s · grounded traffic", fill=(132, 226, 255, 255))
    return np.asarray(image)


class TrafficVehicle:
    def __init__(
        self,
        stage,
        asset: Path,
        index: int,
        lane_y: float,
        direction: float,
        start_x: float,
        distance: float,
        speed: float,
    ) -> None:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self.index = index
        self.lane_y = lane_y
        self.direction = direction
        self.start_x = start_x
        self.distance = distance
        self.speed = speed
        self.progress = 0.0
        self.current_speed = 0.0
        self.last_timestamp_s: float | None = None
        self.last_ground_z: float | None = None
        self.root_path = f"/UrbanVerseTraffic/Vehicle_{index:02d}"
        self.root = stage.DefinePrim(self.root_path, "Xform")
        self.root.GetPayloads().AddPayload(str(asset))
        self.root.Load()
        descendants = list(Usd.PrimRange(self.root))
        mesh_count = sum(item.IsA(UsdGeom.Mesh) for item in descendants)
        if mesh_count < 1:
            raise RuntimeError(f"vehicle {index} payload composed no mesh: {asset}")
        for item in descendants:
            if item.HasAPI(UsdPhysics.RigidBodyAPI):
                item.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if item.HasAPI(UsdPhysics.CollisionAPI):
                item.RemoveAPI(UsdPhysics.CollisionAPI)
        xformable = UsdGeom.Xformable(self.root)
        xformable.ClearXformOpOrder()
        self.translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "traffic")
        self.orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "traffic")
        local_range = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
        ).ComputeLocalBound(self.root).ComputeAlignedRange()
        self.local_center = np.asarray(local_range.GetMidpoint(), dtype=np.float64)
        self.local_size = np.asarray(local_range.GetSize(), dtype=np.float64)
        if not np.all(np.isfinite(self.local_size)) or float(np.min(self.local_size)) <= 0.1:
            raise RuntimeError(f"vehicle {index} has invalid local bounds: {self.local_size.tolist()}")
        yaw = 0.0 if direction > 0.0 else math.pi
        yaw_quat = np.asarray([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
        self.quat_xyzw = multiply_quaternions_xyzw(yaw_quat, BASE_QUAT_XYZW)
        self.orient_op.Set(Gf.Quatd(float(self.quat_xyzw[3]), Gf.Vec3d(*self.quat_xyzw[:3].tolist())))

        proxy_path = f"/UrbanVerseTraffic/CollisionProxy_{index:02d}"
        proxy = UsdGeom.Cube.Define(stage, proxy_path)
        proxy.CreateSizeAttr(1.0)
        proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        proxy_xform = UsdGeom.Xformable(proxy.GetPrim())
        proxy_xform.ClearXformOpOrder()
        self.proxy_translate_op = proxy_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        proxy_xform.AddScaleOp().Set(Gf.Vec3d(*VEHICLE_SIZE_XYZ.tolist()))
        self.states: list[dict[str, Any]] = []

    def update(self, timestamp_s: float, scene_query, commanded_speed: float) -> dict[str, Any]:
        import carb
        from pxr import Gf

        dt = 0.0 if self.last_timestamp_s is None else max(0.0, timestamp_s - self.last_timestamp_s)
        self.last_timestamp_s = timestamp_s
        remaining = max(0.0, self.distance - self.progress)
        target_speed = min(max(0.0, commanded_speed), self.speed) if remaining > 1.0e-6 else 0.0
        rate = 1.0 if target_speed >= self.current_speed else 2.0
        self.current_speed += float(np.clip(target_speed - self.current_speed, -rate * dt, rate * dt))
        advance = min(remaining, self.current_speed * dt)
        self.progress += advance
        if self.progress >= self.distance - 1.0e-6:
            self.progress = self.distance
            self.current_speed = 0.0
        x = self.start_x + self.direction * self.progress
        ray = scene_query.raycast_closest(
            carb.Float3(float(x), float(self.lane_y), 8.0), carb.Float3(0.0, 0.0, -1.0), 20.0
        )
        if not isinstance(ray, dict) or not bool(ray.get("hit", False)) or ray.get("position") is None:
            raise RuntimeError(f"vehicle {self.index} ground ray missed at {(x, self.lane_y)}")
        hit_text = f"{ray.get('collision', '')} {ray.get('rigidBody', '')}".lower()
        if "vehicle_" in hit_text or "/urbanversetraffic" in hit_text:
            raise RuntimeError(f"vehicle {self.index} ground ray hit a vehicle instead of road: {hit_text}")
        ground_z = float(ray["position"][2])
        self.last_ground_z = ground_z
        center_z = ground_z + VEHICLE_HALF_HEIGHT
        center = np.asarray([x, self.lane_y, center_z], dtype=np.float64)
        rotated_center = rotate_vector_xyzw(self.quat_xyzw, self.local_center)
        root_position = center - rotated_center + ROOT_CORRECTION_XYZ
        self.translate_op.Set(Gf.Vec3d(*root_position.tolist()))
        self.proxy_translate_op.Set(Gf.Vec3d(*center.tolist()))
        state = {
            "vehicle_id": self.index,
            "timestamp_s": timestamp_s,
            "center_xyz": center.tolist(),
            "velocity_xy": [self.direction * self.current_speed, 0.0],
            "heading_rad": 0.0 if self.direction > 0.0 else math.pi,
            "moving": self.current_speed > 1.0e-4,
            "progress_m": self.progress,
            "commanded_speed_mps": target_speed,
            "ground_z": ground_z,
            "ground_collision": str(ray.get("collision") or ""),
            "visual_root_xyz": root_position.tolist(),
        }
        self.states.append(state)
        return state

    def current_center_xy(self) -> np.ndarray:
        return np.asarray(
            [self.start_x + self.direction * self.progress, self.lane_y], dtype=np.float64
        )


def contact_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_width = 480
    thumb_height = round(images[0].height * thumb_width / images[0].width)
    canvas = Image.new("RGB", (thumb_width * len(images), thumb_height), (20, 20, 20))
    for index, image in enumerate(images):
        canvas.paste(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (index * thumb_width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    args = parse_args()
    source = args.usd.resolve()
    source_tar = args.tar.resolve()
    asset = args.vehicle_asset.resolve()
    run_dir = args.run_dir.resolve()
    metadata = run_dir / "metadata"
    captures = run_dir / "captures"
    visualizations = run_dir / "visualizations"
    for directory in (metadata, captures, visualizations):
        directory.mkdir(parents=True, exist_ok=True)
    if QUALIFIED_ASSET_ID not in asset.name:
        raise RuntimeError(f"unqualified vehicle asset: {asset}")
    hashes_before = {"usd": sha256(source), "tar": sha256(source_tar), "vehicle_asset": sha256(asset)}
    wrapper = run_dir / "scene10_straight_traffic_wrapper.usda"
    wrapper.write_text(f'#usda 1.0\n(\n    subLayers = [@{source.as_posix()}@]\n)\n', encoding="utf-8")
    environment = {
        "timestamp": now(), "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "isaac_sim": package_version("isaacsim"),
        "replicator": package_version("isaacsim-replicator"), "kit": package_version("isaacsim-kernel"),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "command_line": sys.argv, "physical_gpu_index": args.gpu, "nvidia_smi": command_output(["nvidia-smi"]),
        "renderer": "RayTracedLighting", "resolution": [args.width, args.height],
        "camera": "/UrbanVerseTraffic/Camera", "source_usd": str(source), "source_tar": str(source_tar),
        "vehicle_asset": str(asset), "source_asset_hashes": hashes_before, "wrapper_usd": str(wrapper),
    }
    write_json(metadata / "environment.json", environment)
    summary: dict[str, Any] = {"status": "running", "started_at": now()}
    write_json(metadata / "summary.json", summary)
    app = None
    annotator = None
    render_product = None
    timeline = None
    started = time.perf_counter()
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({
            "headless": True, "renderer": "RayTracedLighting", "width": args.width, "height": args.height,
            "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False, "max_gpu_count": 1,
            "create_new_stage": False,
            "extra_args": ["--ext-folder", str(args.ext_folder.resolve()), "--/renderer/multiGpu/enabled=false", "--/app/window/hideUi=1"],
        }, experience=str(args.experience.resolve()))
        import carb
        import cv2
        import omni.kit.app
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
        for _ in range(4):
            app.update()
        context = omni.usd.get_context()
        if not context.open_stage(str(wrapper)):
            raise RuntimeError(f"could not open stage: {wrapper}")
        wait_stage(context, app)
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetRootLayer())
        carb.settings.get_settings().set("/rtx/post/tonemap/exposure", 0.0)

        static_vehicle_inventory: list[dict[str, Any]] = []
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
        )
        for source_vehicle in stage.Traverse():
            path = str(source_vehicle.GetPath())
            if path.startswith("/UrbanVerseTraffic") or "vehicle_private_vehicle" not in source_vehicle.GetName():
                continue
            world_range = bbox_cache.ComputeWorldBound(source_vehicle).ComputeAlignedRange()
            center = np.asarray(world_range.GetMidpoint(), dtype=np.float64)
            size = np.asarray(world_range.GetSize(), dtype=np.float64)
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)) or float(np.min(size)) <= 0.01:
                continue
            static_vehicle_inventory.append({
                "path": path,
                "center_xyz": center.tolist(),
                "size_xyz": size.tolist(),
                "min_xyz": np.asarray(world_range.GetMin(), dtype=np.float64).tolist(),
                "max_xyz": np.asarray(world_range.GetMax(), dtype=np.float64).tolist(),
            })
        write_json(metadata / "static_vehicle_inventory.json", static_vehicle_inventory)
        specs = [
            (TRAFFIC_LANE_Y, 1.0, -629.0, 10.0, 1.0),
            (TRAFFIC_LANE_Y, 1.0, -617.0, 6.0, 0.8),
        ]
        vehicles = [TrafficVehicle(stage, asset, index, *spec) for index, spec in enumerate(specs)]
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(24):
            app.update()
        scene_query = omni.physx.get_physx_scene_query_interface()

        target = Gf.Vec3d(-619.0, TRAFFIC_LANE_Y, 1.0)
        eye = target + Gf.Vec3d(14.0, -22.0, 24.0)
        camera = UsdGeom.Camera.Define(stage, "/UrbanVerseTraffic/Camera")
        camera.CreateFocalLengthAttr(25.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100000.0))
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.ClearXformOpOrder()
        camera_xform.AddTransformOp().Set(Gf.Matrix4d(1.0).SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
        render_product = rep.create.render_product(camera.GetPath(), (args.width, args.height), force_new=True)
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        for _ in range(30):
            app.update()
        rep.orchestrator.step(rt_subframes=2)

        frame_count = int(round(args.duration * args.fps))
        video_path = captures / "scene10_straight_traffic.webm"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"VP90"), args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError("could not open VP9 writer")
        key_indices = sorted(set([0, frame_count // 3, 2 * frame_count // 3, frame_count - 1]))
        key_paths: list[Path] = []
        minimum_same_lane_gap = math.inf
        minimum_dynamic_edge_gap = math.inf
        minimum_dynamic_static_clearance = math.inf
        minimum_wheel_ground_gap = math.inf
        maximum_wheel_ground_gap = -math.inf
        static_collision_events: list[dict[str, Any]] = []
        dynamic_collision_events: list[dict[str, Any]] = []
        ground_contact_failures: list[dict[str, Any]] = []
        dynamic_bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
        )

        def static_obstacle_speed_limit(vehicle: TrafficVehicle) -> tuple[float, float | None]:
            center = vehicle.current_center_xy()
            nearest_gap = math.inf
            for static in static_vehicle_inventory:
                static_center = np.asarray(static["center_xyz"][:2], dtype=np.float64)
                static_size = np.asarray(static["size_xyz"][:2], dtype=np.float64)
                if float(np.max(static_size)) > 15.0:
                    continue
                lateral_gap = abs(center[1] - static_center[1]) - VEHICLE_SIZE_XYZ[1] / 2.0 - static_size[1] / 2.0
                if lateral_gap > 0.45:
                    continue
                if vehicle.direction > 0.0:
                    gap = static_center[0] - static_size[0] / 2.0 - (center[0] + VEHICLE_SIZE_XYZ[0] / 2.0)
                else:
                    gap = center[0] - VEHICLE_SIZE_XYZ[0] / 2.0 - (static_center[0] + static_size[0] / 2.0)
                if gap >= 0.0:
                    nearest_gap = min(nearest_gap, float(gap))
            if not math.isfinite(nearest_gap):
                return vehicle.speed, None
            if nearest_gap <= 2.8:
                return 0.0, nearest_gap
            if nearest_gap < 8.0:
                return min(vehicle.speed, 0.32 * (nearest_gap - 2.8)), nearest_gap
            return vehicle.speed, nearest_gap

        state_path = metadata / "traffic_states.jsonl"
        try:
            with state_path.open("w", encoding="utf-8") as stream:
                for frame_index in range(frame_count):
                    timestamp_s = frame_index / args.fps
                    leader_limit, leader_static_gap = static_obstacle_speed_limit(vehicles[1])
                    leader_state = vehicles[1].update(timestamp_s, scene_query, leader_limit)
                    follower_xy = vehicles[0].current_center_xy()
                    leader_xy = np.asarray(leader_state["center_xyz"][:2], dtype=np.float64)
                    follower_edge_gap = leader_xy[0] - follower_xy[0] - VEHICLE_SIZE_XYZ[0]
                    if follower_edge_gap <= 3.5:
                        following_limit = 0.0
                    elif follower_edge_gap < 9.0:
                        following_limit = min(
                            vehicles[0].speed,
                            float(leader_state["velocity_xy"][0]) + 0.30 * (follower_edge_gap - 3.5),
                        )
                    else:
                        following_limit = vehicles[0].speed
                    follower_static_limit, follower_static_gap = static_obstacle_speed_limit(vehicles[0])
                    follower_state = vehicles[0].update(
                        timestamp_s, scene_query, min(following_limit, follower_static_limit)
                    )
                    follower_state["following_edge_gap_m"] = follower_edge_gap
                    follower_state["static_obstacle_gap_m"] = follower_static_gap
                    leader_state["static_obstacle_gap_m"] = leader_static_gap
                    states = [follower_state, leader_state]
                    rep.orchestrator.step(rt_subframes=1)
                    visible_ranges: list[tuple[np.ndarray, np.ndarray]] = []
                    for vehicle, state in zip(vehicles, states):
                        dynamic_bbox_cache.Clear()
                        world_range = dynamic_bbox_cache.ComputeWorldBound(vehicle.root).ComputeAlignedRange()
                        visual_min = np.asarray(world_range.GetMin(), dtype=np.float64)
                        visual_max = np.asarray(world_range.GetMax(), dtype=np.float64)
                        visible_ranges.append((visual_min, visual_max))
                        state["visual_bbox_min_xyz"] = visual_min.tolist()
                        state["visual_bbox_max_xyz"] = visual_max.tolist()
                        state["visual_bbox_center_xyz"] = ((visual_min + visual_max) / 2.0).tolist()
                        visual_min_z = float(visual_min[2])
                        wheel_ground_gap = visual_min_z - float(state["ground_z"])
                        state["wheel_ground_gap_m"] = wheel_ground_gap
                        minimum_wheel_ground_gap = min(minimum_wheel_ground_gap, wheel_ground_gap)
                        maximum_wheel_ground_gap = max(maximum_wheel_ground_gap, wheel_ground_gap)
                        if abs(wheel_ground_gap) > 0.03 and len(ground_contact_failures) < 100:
                            ground_contact_failures.append({
                                "frame": frame_index, "timestamp_s": timestamp_s,
                                "vehicle_id": state["vehicle_id"], "wheel_ground_gap_m": wheel_ground_gap,
                            })
                    visible_centers = [(bounds[0] + bounds[1]) / 2.0 for bounds in visible_ranges]
                    center_gap = abs(float(visible_centers[1][0] - visible_centers[0][0]))
                    minimum_same_lane_gap = min(minimum_same_lane_gap, center_gap)
                    left, right = sorted(visible_ranges, key=lambda bounds: float(bounds[0][0]))
                    edge_gap = float(right[0][0] - left[1][0])
                    minimum_dynamic_edge_gap = min(minimum_dynamic_edge_gap, edge_gap)
                    if edge_gap < 0.0 and len(dynamic_collision_events) < 100:
                        dynamic_collision_events.append({
                            "frame": frame_index, "timestamp_s": timestamp_s, "edge_gap_m": edge_gap
                        })
                    for state, (visual_min, visual_max) in zip(states, visible_ranges):
                        dynamic_xy = (visual_min[:2] + visual_max[:2]) / 2.0
                        dynamic_half = (visual_max[:2] - visual_min[:2]) / 2.0
                        for static in static_vehicle_inventory:
                            static_xy = np.asarray(static["center_xyz"][:2], dtype=np.float64)
                            static_half = np.asarray(static["size_xyz"][:2], dtype=np.float64) / 2.0
                            axis_clearance = np.abs(dynamic_xy - static_xy) - dynamic_half - static_half
                            positive = np.maximum(axis_clearance, 0.0)
                            clearance = (
                                float(np.linalg.norm(positive))
                                if np.any(positive > 0.0)
                                else float(np.max(axis_clearance))
                            )
                            minimum_dynamic_static_clearance = min(minimum_dynamic_static_clearance, clearance)
                            if clearance < 0.0 and len(static_collision_events) < 100:
                                static_collision_events.append({
                                    "frame": frame_index, "timestamp_s": timestamp_s,
                                    "dynamic_vehicle_id": state["vehicle_id"],
                                    "static_vehicle_path": static["path"], "clearance_m": clearance,
                                })
                    raw = rgb_array(annotator)
                    frame = annotate(raw, timestamp_s)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    stream.write(json.dumps({"frame": frame_index, "timestamp_s": timestamp_s, "vehicles": states}) + "\n")
                    if frame_index in key_indices:
                        path = captures / f"frame_{frame_index:04d}.png"
                        Image.fromarray(frame).save(path)
                        key_paths.append(path)
        finally:
            writer.release()

        capture = cv2.VideoCapture(str(video_path))
        video_validation = {
            "opened": bool(capture.isOpened()), "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))), "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)), "file_size_bytes": video_path.stat().st_size,
        }
        capture.release()
        hashes_after = {"usd": sha256(source), "tar": sha256(source_tar), "vehicle_asset": sha256(asset)}
        if hashes_after != hashes_before:
            raise RuntimeError("source asset hash changed")
        if video_validation["frame_count"] != frame_count:
            raise RuntimeError(f"video validation failed: {video_validation}")
        contact_path = visualizations / "straight_traffic_contact_sheet.png"
        contact_sheet(key_paths, contact_path)
        if minimum_dynamic_edge_gap < 3.0:
            raise RuntimeError(f"unsafe dynamic edge gap: {minimum_dynamic_edge_gap}")
        write_json(metadata / "dynamic_static_collision_events.json", static_collision_events)
        write_json(metadata / "dynamic_dynamic_collision_events.json", dynamic_collision_events)
        write_json(metadata / "ground_contact_failures.json", ground_contact_failures)
        if static_collision_events:
            raise RuntimeError(
                f"dynamic traffic intersects static vehicles: {len(static_collision_events)} sampled events, "
                f"minimum clearance={minimum_dynamic_static_clearance:.3f}m"
            )
        if dynamic_collision_events:
            raise RuntimeError(f"dynamic vehicles overlap: {len(dynamic_collision_events)} sampled events")
        if ground_contact_failures:
            raise RuntimeError(
                f"vehicle visual ground contact failed: {len(ground_contact_failures)} sampled events, "
                f"range=[{minimum_wheel_ground_gap:.4f}, {maximum_wheel_ground_gap:.4f}]m"
            )
        summary.update({
            "status": "success", "completed_at": now(), "wall_time_s": time.perf_counter() - started,
            "scene": "scene_10_cbd_cross_intersection_diverse_obstacles", "go2_present": False,
            "traffic": {"vehicle_count": len(vehicles), "lane_count": 1, "motion": "finite straight kinematic whole-body",
                        "minimum_same_lane_center_gap_m": minimum_same_lane_gap,
                        "minimum_dynamic_edge_gap_m": minimum_dynamic_edge_gap,
                        "visual_asset": str(asset), "collision_proxy": "aligned invisible kinematic box per vehicle",
                        "vehicle_size_xyz": VEHICLE_SIZE_XYZ.tolist(), "wheel_rotation": False, "ackermann": False,
                        "retained_static_vehicle_count": len(static_vehicle_inventory),
                        "minimum_dynamic_static_clearance_m": minimum_dynamic_static_clearance,
                        "ground_height_source": "per-frame downward PhysX raycast",
                        "minimum_wheel_ground_gap_m": minimum_wheel_ground_gap,
                        "maximum_wheel_ground_gap_m": maximum_wheel_ground_gap,
                        "runtime_car_following": True, "runtime_static_obstacle_braking": True},
            "validation": {"qualified_asset": True, "heading_matches_velocity": True,
                           "ground_raycast_aligned": True,
                           "source_modified": False, "no_same_lane_overlap": True,
                           "all_static_vehicles_retained": True, "no_dynamic_static_overlap": True},
            "video": {"path": str(video_path), **video_validation}, "keyframes": [str(path) for path in key_paths],
            "visualizations": [str(contact_path)], "source_hashes_before": hashes_before, "source_hashes_after": hashes_after,
        })
        write_json(metadata / "summary.json", summary)
        (run_dir / "run_log.txt").write_text(
            f"STATUS success\nVEHICLES {len(vehicles)}\nLANES 1\nMIN_SAME_LANE_GAP_M {minimum_same_lane_gap:.3f}\n"
            f"VIDEO {video_path}\nGO2_PRESENT false\nSOURCE_MODIFIED false\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "success", "run_dir": str(run_dir), "video": str(video_path)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        summary.update({"status": "failed", "completed_at": now(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        write_json(metadata / "summary.json", summary)
        (run_dir / "run_log.txt").write_text(summary["traceback"], encoding="utf-8")
        traceback.print_exc()
        return 1
    finally:
        if timeline is not None:
            try:
                timeline.stop()
            except Exception:
                pass
        if annotator is not None:
            try:
                annotator.detach()
            except Exception:
                pass
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
        if app is not None:
            try:
                app.close(wait_for_replicator=False)
            except TypeError:
                app.close()


if __name__ == "__main__":
    raise SystemExit(main())
