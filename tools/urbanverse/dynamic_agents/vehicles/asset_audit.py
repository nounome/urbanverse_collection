#!/usr/bin/env python3
"""Read-only vehicle asset audit for CraftBench scene_10.

The audit opens the source USD without authoring changes, discovers native
vehicle roots, measures their visible geometry in world space, and writes a
machine-readable inventory plus an annotated top-down OBB overview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--ext-folder", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=1500)
    return parser.parse_args()


def wait_stage(context, app, max_updates: int = 1800) -> None:
    for _ in range(max_updates):
        app.update()
        try:
            status = tuple(context.get_stage_loading_status())
        except Exception:
            status = ()
        if not status or (len(status) >= 3 and int(status[-1]) == 0):
            return
    raise RuntimeError(f"stage loading timed out: {context.get_stage_loading_status()}")


def convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(points) <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin, left, right) -> float:
        return float(np.cross(left - origin, right - origin))

    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def minimum_rectangle(points: np.ndarray) -> dict[str, Any]:
    hull = convex_hull(points)
    if len(hull) < 2:
        raise ValueError("not enough projected points for OBB")
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), math.pi / 2.0))
    best = None
    for angle in angles:
        axis_x = np.asarray([math.cos(angle), math.sin(angle)])
        axis_y = np.asarray([-math.sin(angle), math.cos(angle)])
        projected_x = hull @ axis_x
        projected_y = hull @ axis_y
        lo_x, hi_x = float(projected_x.min()), float(projected_x.max())
        lo_y, hi_y = float(projected_y.min()), float(projected_y.max())
        area = (hi_x - lo_x) * (hi_y - lo_y)
        candidate = (area, angle, axis_x, axis_y, lo_x, hi_x, lo_y, hi_y)
        if best is None or area < best[0]:
            best = candidate
    assert best is not None
    _, angle, axis_x, axis_y, lo_x, hi_x, lo_y, hi_y = best
    extent_x, extent_y = hi_x - lo_x, hi_y - lo_y
    center = axis_x * ((lo_x + hi_x) / 2.0) + axis_y * ((lo_y + hi_y) / 2.0)
    if extent_y > extent_x:
        extent_x, extent_y = extent_y, extent_x
        axis_x, axis_y = axis_y, -axis_x
        angle += math.pi / 2.0
    half_x, half_y = extent_x / 2.0, extent_y / 2.0
    corners = [
        center - axis_x * half_x - axis_y * half_y,
        center + axis_x * half_x - axis_y * half_y,
        center + axis_x * half_x + axis_y * half_y,
        center - axis_x * half_x + axis_y * half_y,
    ]
    return {
        "center_xy": center.tolist(),
        "length_width": [float(extent_x), float(extent_y)],
        "heading_deg": float(math.degrees(math.atan2(axis_x[1], axis_x[0]))),
        "forward_axis_xy_unsigned": axis_x.tolist(),
        "corners_xy": [corner.tolist() for corner in corners],
        "hull_vertex_count": int(len(hull)),
    }


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def overview(records: list[dict[str, Any]], output: Path, width: int, height: int) -> None:
    valid = [record for record in records if record.get("obb")]
    points = np.asarray([corner for record in valid for corner in record["obb"]["corners_xy"]], dtype=np.float64)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    title_h, footer_h, margin = 125, 92, 55
    map_w, map_h = width - 2 * margin, height - title_h - footer_h
    scale = min(map_w / span[0], map_h / span[1]) * 0.94
    used_w, used_h = span[0] * scale, span[1] * scale
    origin_x = margin + (map_w - used_w) / 2.0
    origin_y = title_h + (map_h - used_h) / 2.0

    def project(point) -> tuple[float, float]:
        return (
            float(origin_x + (point[0] - lo[0]) * scale),
            float(origin_y + used_h - (point[1] - lo[1]) * scale),
        )

    image = Image.new("RGB", (width, height), (19, 25, 32))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((margin, title_h, width - margin, height - footer_h), fill=(222, 226, 226, 255))
    title = font(34, True)
    body = font(20)
    small = font(16)
    draw.text((margin, 28), "scene_10 车辆资产审计 · 可见几何 OBB 俯视分布", font=title, fill=(245, 247, 249, 255))
    translation_only_count = sum(record["quality"]["status"] == "translation_only" for record in valid)
    rejected_count = sum(record["quality"]["status"] == "reject" for record in records)
    draw.text((margin, 79), f"车辆根节点 {len(records)} · 可整车平移 {translation_only_count} · 剔除 {rejected_count}", font=body, fill=(177, 190, 201, 255))
    for index, record in enumerate(valid, start=1):
        status = record["quality"]["status"]
        color = {
            "full_pose_candidate": (35, 137, 210, 190),
            "translation_only": (229, 157, 48, 210),
            "reject": (225, 74, 67, 210),
        }[status]
        corners = [project(corner) for corner in record["obb"]["corners_xy"]]
        draw.polygon(corners, fill=(*color[:3], 80), outline=color)
        center = project(record["obb"]["center_xy"])
        draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=color)
        draw.text((center[0] + 5, center[1] - 10), str(index), font=small, fill=(15, 20, 25, 255))
    draw.text((margin, height - 63), "蓝：完整位姿候选    橙：车身正常但远心根节点，仅建议整车平移    红：尺寸/网格异常，剔除", font=body, fill=(220, 226, 231, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    args = parse_args()
    source, source_tar = args.usd.resolve(), args.tar.resolve()
    run_dir = args.run_dir.resolve()
    metadata, captures, visualizations = run_dir / "metadata", run_dir / "captures", run_dir / "visualizations"
    for directory in (metadata, captures, visualizations):
        directory.mkdir(parents=True, exist_ok=False)
    hashes_before = {"usd_sha256": sha256(source), "tar_sha256": sha256(source_tar)}
    environment = {
        "timestamp": now(), "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "isaac_sim": package_version("isaacsim"),
        "replicator": package_version("isaacsim-replicator"), "kit": package_version("isaacsim-kernel"),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]), "command_line": sys.argv,
        "physical_gpu_index": args.gpu,
        "gpu_audit": command_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "renderer": "none; USD geometry audit", "resolution": [args.width, args.height],
        "camera": "none; coordinate-stable synthetic top-down OBB overview",
        "source_usd": str(source), "source_tar": str(source_tar), "source_asset_hashes": hashes_before,
    }
    write_json(metadata / "environment.json", environment)
    write_json(metadata / "summary.json", {"status": "running", "started_at": now()})
    app = None
    started = time.perf_counter()
    try:
        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": True,
                "renderer": "RayTracedLighting",
                "active_gpu": args.gpu,
                "physics_gpu": args.gpu,
                "multi_gpu": False,
                "max_gpu_count": 1,
                "create_new_stage": False,
                "extra_args": [
                    "--ext-folder", str(args.ext_folder.resolve()),
                    "--/renderer/multiGpu/enabled=false",
                    "--/app/window/hideUi=1",
                ],
            },
            experience=str(args.experience.resolve()),
        )
        import omni.kit.app
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
        for _ in range(5):
            app.update()
        context = omni.usd.get_context()
        if not context.open_stage(str(source)):
            raise RuntimeError(f"could not open stage: {source}")
        wait_stage(context, app)
        stage = context.get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        candidates = [prim for prim in stage.Traverse() if "vehicle_private_vehicle" in prim.GetName().lower()]
        candidate_paths = {str(prim.GetPath()) for prim in candidates}
        roots = [prim for prim in candidates if not any(str(prim.GetPath()).startswith(path + "/") for path in candidate_paths if path != str(prim.GetPath()))]
        records: list[dict[str, Any]] = []
        for root in roots:
            root.Load()
            root_path = str(root.GetPath())
            descendants = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
            meshes = [prim for prim in descendants if prim.IsA(UsdGeom.Mesh)]
            root_matrix = xform_cache.GetLocalToWorldTransform(root)
            root_origin = np.asarray(root_matrix.Transform(Gf.Vec3d(0.0)), dtype=np.float64)
            projected_chunks: list[np.ndarray] = []
            mesh_records: list[dict[str, Any]] = []
            for mesh_prim in meshes:
                mesh = UsdGeom.Mesh(mesh_prim)
                points_value = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
                if not points_value:
                    continue
                matrix = xform_cache.GetLocalToWorldTransform(mesh_prim)
                world_points = np.asarray([matrix.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in points_value], dtype=np.float64)
                if len(world_points) > 25000:
                    indices = np.linspace(0, len(world_points) - 1, 25000, dtype=np.int64)
                    projected_chunks.append(world_points[indices, :2])
                else:
                    projected_chunks.append(world_points[:, :2])
                mesh_records.append({"path": str(mesh_prim.GetPath()), "vertex_count": int(len(world_points)), "min_xyz": world_points.min(axis=0).tolist(), "max_xyz": world_points.max(axis=0).tolist()})
            if projected_chunks:
                all_projected = np.concatenate(projected_chunks, axis=0)
                obb = minimum_rectangle(all_projected)
                mesh_mins = np.asarray([record["min_xyz"] for record in mesh_records], dtype=np.float64)
                mesh_maxs = np.asarray([record["max_xyz"] for record in mesh_records], dtype=np.float64)
                world_min = mesh_mins.min(axis=0)
                world_max = mesh_maxs.max(axis=0)
                world_center = (world_min + world_max) / 2.0
                world_size = world_max - world_min
                root_offset = world_center - root_origin
            else:
                obb = None
                world_min = world_max = world_center = world_size = root_offset = None
            asset_refs: list[dict[str, Any]] = []
            for prim_spec in root.GetPrimStack():
                for item in prim_spec.payloadList.GetAddedOrExplicitItems():
                    resolved = Sdf.ComputeAssetPathRelativeToLayer(prim_spec.layer, item.assetPath)
                    asset_refs.append({"authored": item.assetPath, "resolved": resolved, "prim_path": str(item.primPath)})
                for item in prim_spec.referenceList.GetAddedOrExplicitItems():
                    if item.assetPath:
                        resolved = Sdf.ComputeAssetPathRelativeToLayer(prim_spec.layer, item.assetPath)
                        asset_refs.append({"authored": item.assetPath, "resolved": resolved, "prim_path": str(item.primPath), "kind": "reference"})
            issues: list[str] = []
            if not meshes:
                issues.append("no_visible_mesh")
            if root_offset is not None and float(np.linalg.norm(root_offset[:2])) > 10.0:
                issues.append("root_origin_far_from_visible_body")
            if obb:
                length, width = obb["length_width"]
                if not (2.2 <= length <= 10.0 and 1.1 <= width <= 4.0 and 0.7 <= world_size[2] <= 5.0):
                    issues.append("implausible_vehicle_dimensions")
                if length / max(width, 1.0e-6) < 1.15:
                    issues.append("ambiguous_forward_axis")
            if len(meshes) > 250:
                issues.append("unusually_complex_mesh_hierarchy")
            collision_count = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in descendants)
            rigid_count = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in descendants)
            hard_reject = any(issue in issues for issue in ("no_visible_mesh", "implausible_vehicle_dimensions"))
            if hard_reject:
                quality_status = "reject"
            elif "root_origin_far_from_visible_body" in issues:
                quality_status = "translation_only"
            else:
                quality_status = "full_pose_candidate"
            records.append({
                "path": root_path, "name": root.GetName(), "type_name": root.GetTypeName(), "active": root.IsActive(), "loaded": root.IsLoaded(),
                "asset_references": asset_refs, "descendant_count": len(descendants), "mesh_count": len(meshes),
                "collision_api_count": collision_count, "rigid_body_api_count": rigid_count,
                "root_has_rigid_body_api": root.HasAPI(UsdPhysics.RigidBodyAPI),
                "kinematic_enabled": root.GetAttribute("physics:kinematicEnabled").Get(),
                "root_world_origin_xyz": root_origin.tolist(),
                "visible_world_aabb_min_xyz": world_min.tolist() if world_min is not None else None,
                "visible_world_aabb_max_xyz": world_max.tolist() if world_max is not None else None,
                "visible_world_aabb_center_xyz": world_center.tolist() if world_center is not None else None,
                "visible_world_aabb_size_xyz": world_size.tolist() if world_size is not None else None,
                "root_to_visible_center_offset_xyz": root_offset.tolist() if root_offset is not None else None,
                "visible_bottom_z": float(world_min[2]) if world_min is not None else None,
                "obb": obb, "mesh_records": mesh_records,
                "quality": {
                    "status": quality_status,
                    "issues": issues,
                    "recommended_motion": "whole-body translation only" if quality_status == "translation_only" else ("exclude" if quality_status == "reject" else "pose control candidate"),
                },
            })
        records.sort(key=lambda record: record["path"])
        write_json(metadata / "vehicle_inventory.json", records)
        map_path = visualizations / "scene10_vehicle_obb_overview.png"
        overview(records, map_path, args.width, args.height)
        Image.open(map_path).save(captures / "scene10_vehicle_obb_overview_rgb.png")
        issue_counts: dict[str, int] = {}
        for record in records:
            for issue in record["quality"]["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        hashes_after = {"usd_sha256": sha256(source), "tar_sha256": sha256(source_tar)}
        if hashes_after != hashes_before:
            raise RuntimeError("source asset hash changed during read-only audit")
        summary = {
            "status": "passed", "completed_at": now(), "elapsed_s": time.perf_counter() - started,
            "scene": "scene_10_cbd_cross_intersection_diverse_obstacles", "vehicle_root_count": len(records),
            "full_pose_candidate_count": sum(record["quality"]["status"] == "full_pose_candidate" for record in records),
            "translation_only_candidate_count": sum(record["quality"]["status"] == "translation_only" for record in records),
            "reject_count": sum(record["quality"]["status"] == "reject" for record in records),
            "issue_counts": issue_counts, "source_modified": False,
            "claim_scope": "read-only authored hierarchy and visible-geometry bounds audit; OBB forward direction is unsigned",
        }
        write_json(metadata / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failure = {"status": "failed", "completed_at": now(), "elapsed_s": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        write_json(metadata / "summary.json", failure)
        print(failure["traceback"], file=sys.stderr)
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
