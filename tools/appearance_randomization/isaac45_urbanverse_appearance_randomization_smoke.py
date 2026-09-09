#!/usr/bin/env python3
"""RTX 3090 + Isaac Sim 4.5 UrbanVerse baseline and appearance evidence.

One invocation evaluates one source USD in one fresh Isaac process.  The source
USD is sublayered into a run-local wrapper, and every variant is authored only
in the anonymous session layer.  No source asset is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
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


VARIANTS: list[dict[str, Any]] = [
    {
        "id": "baseline",
        "kind": "baseline",
        "seed_offset": 0,
        "tests": ["真实纹理", "材质", "阴影", "反射"],
        "display_name_zh": "原始外观基准",
        "explanation_zh": "不施加随机化，用来和后续增强档位直接比较。",
    },
    {
        "id": "light_bright_noon",
        "kind": "light",
        "profile": "bright_noon",
        "seed_offset": 11,
        "tests": ["光照变化", "时间", "阴影", "光照强度随机化", "光照方向随机化"],
        "display_name_zh": "强光·冷白正午",
        "explanation_zh": "提高天空光和太阳光，并改变太阳方向，突出亮度和阴影差异。",
    },
    {
        "id": "light_warm_sunset",
        "kind": "light",
        "profile": "warm_sunset",
        "seed_offset": 17,
        "tests": ["光照变化", "时间", "阴影", "光照强度随机化", "光照方向随机化"],
        "display_name_zh": "弱光·暖色日落",
        "explanation_zh": "降低环境光、使用低角度暖色太阳，形成和正午相反的对照。",
    },
    {
        "id": "weather_overcast_cool_proxy",
        "kind": "weather",
        "profile": "overcast_cool",
        "seed_offset": 23,
        "tests": ["天气", "时间", "天空随机化"],
        "display_name_zh": "天气代理·冷色阴天",
        "explanation_zh": "用冷灰色 DomeLight 模拟阴天观感；不是完整天气系统。",
    },
    {
        "id": "weather_storm_dark_proxy",
        "kind": "weather",
        "profile": "storm_dark",
        "seed_offset": 29,
        "tests": ["天气", "时间", "天空随机化"],
        "display_name_zh": "天气代理·深蓝风暴",
        "explanation_zh": "用深蓝天空光和负曝光形成强烈风暴前观感；没有雨、雾或湿地交互。",
    },
    {
        "id": "ground_material_warm_rough",
        "kind": "material",
        "target_kind": "ground",
        "profile": "warm_rough",
        "seed_offset": 37,
        "tests": ["地面材质随机化"],
        "display_name_zh": "地面材质·暖橙粗糙",
        "explanation_zh": "把选中的源场景地面 prim 绑定为高粗糙度暖橙材质。",
    },
    {
        "id": "ground_material_cool_glossy",
        "kind": "material",
        "target_kind": "ground",
        "profile": "cool_glossy",
        "seed_offset": 43,
        "tests": ["地面材质随机化"],
        "display_name_zh": "地面材质·冷蓝光滑",
        "explanation_zh": "把相同类别地面 prim 改为冷蓝、较低粗糙度材质，便于和暖色档比较。",
    },
    {
        "id": "ground_material_neutral_preview_compat",
        "kind": "material",
        "target_kind": "ground",
        "profile": "neutral_preview_compat",
        "seed_offset": 44,
        "tests": ["地面材质兼容性修复"],
        "display_name_zh": "地面兼容档·中性 PreviewSurface",
        "explanation_zh": "用中性、非金属、高粗糙度 UsdPreviewSurface 覆盖被审计的源地面 prim，作为 MDL/投影不兼容时的非破坏性兼容回退。",
    },
    {
        "id": "ground_material_dark_asphalt_replacement",
        "kind": "material",
        "target_kind": "ground",
        "profile": "dark_asphalt_replacement",
        "seed_offset": 46,
        "tests": ["地面材质替换"],
        "display_name_zh": "地面训练档·深色粗糙沥青",
        "explanation_zh": "把同一批被审计的源地面 prim 改为深色高粗糙度材质，用来验证降低泛白后的训练外观；不修改原始 USD。",
    },
    {
        "id": "road_surface_dark_asphalt_overlay_proxy",
        "kind": "road_overlay",
        "seed_offset": 45,
        "tests": ["地面材质随机化"],
        "display_name_zh": "道路表面·深色沥青覆盖层",
        "explanation_zh": "在相机前方可见道路高度增加深色高粗糙度表面覆盖层，用于明确展示道路外观差异。",
    },
    {
        "id": "object_material_magenta",
        "kind": "material",
        "target_kind": "object",
        "profile": "magenta",
        "seed_offset": 47,
        "tests": ["物体颜色/纹理随机化"],
        "display_name_zh": "物体材质·洋红",
        "explanation_zh": "对选中的源物体 prim 施加高饱和洋红颜色材质。",
    },
    {
        "id": "object_material_cyan",
        "kind": "material",
        "target_kind": "object",
        "profile": "cyan",
        "seed_offset": 53,
        "tests": ["物体颜色/纹理随机化"],
        "display_name_zh": "物体材质·青色",
        "explanation_zh": "对同类源物体 prim 施加高饱和青色材质；仍不是纹理库替换。",
    },
    {
        "id": "camera_wide_bright",
        "kind": "camera",
        "profile": "wide_bright",
        "seed_offset": 59,
        "tests": ["相机参数噪声"],
        "display_name_zh": "相机·广角亮曝光",
        "explanation_zh": "明显减小焦距、扩大 aperture，并加入位姿扰动和正曝光。",
    },
    {
        "id": "camera_tele_dark",
        "kind": "camera",
        "profile": "tele_dark",
        "seed_offset": 61,
        "tests": ["相机参数噪声"],
        "display_name_zh": "相机·长焦暗曝光",
        "explanation_zh": "明显增大焦距、缩小 aperture，并加入相反位姿扰动和负曝光。",
    },
    {
        "id": "reflection_mirror_proxy",
        "kind": "reflection",
        "profile": "mirror",
        "seed_offset": 67,
        "tests": ["反射"],
        "display_name_zh": "反射代理·镜面银球",
        "explanation_zh": "在相机前加入大尺寸低粗糙度金属球，用于验证 renderer 镜面反射能力。",
    },
    {
        "id": "reflection_rough_gold_proxy",
        "kind": "reflection",
        "profile": "rough_gold",
        "seed_offset": 71,
        "tests": ["反射"],
        "display_name_zh": "反射代理·粗糙金球",
        "explanation_zh": "加入较粗糙的金色金属球，和镜面银球形成反射强度对照。",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("audit", "baseline", "appearance"), required=True)
    p.add_argument("--family", choices=("craftbench", "training"), required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--usd", type=Path, required=True)
    p.add_argument("--tar", type=Path)
    p.add_argument("--camera-path", default="auto")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--experience", type=Path, required=True)
    p.add_argument("--wrapper-template", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--settle-frames", type=int, default=20)
    p.add_argument("--rt-subframes", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument(
        "--variant-filter",
        default="all",
        help="Comma-separated appearance variant ids; baseline is added automatically.",
    )
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def matrix_rows(matrix) -> list[list[float]]:
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        return None


def image_metrics(rgb: np.ndarray, depth: np.ndarray | None) -> dict[str, Any]:
    rgb = np.asarray(rgb, dtype=np.uint8)[..., :3]
    black = np.all(rgb == 0, axis=2)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    channel_span = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    washed_out = (luma >= 190.0) & (channel_span <= 18)
    lower = luma[rgb.shape[0] // 2 :, :]
    gy, gx = np.gradient(lower.astype(np.float32))
    result: dict[str, Any] = {
        "shape": list(rgb.shape),
        "pure_black_definition": "all three uint8 RGB channels equal 0",
        "pure_black_ratio": float(black.mean()),
        "lower_half_pure_black_ratio": float(black[rgb.shape[0] // 2 :, :].mean()),
        "overexposed_definition": "all RGB channels >= 250",
        "overexposed_ratio": float(np.all(rgb >= 250, axis=2).mean()),
        "near_saturated_definition": "at least one RGB channel >= 250",
        "near_saturated_ratio": float(np.any(rgb >= 250, axis=2).mean()),
        "mean_luma": float(luma.mean()),
        "washed_out_definition": "luma >= 190 and RGB channel span <= 18",
        "washed_out_ratio": float(washed_out.mean()),
        "lower_half_washed_out_ratio": float(washed_out[rgb.shape[0] // 2 :, :].mean()),
        "lower_half_luma_mean": float(lower.mean()),
        "lower_half_gradient_mean": float(np.mean(np.hypot(gx, gy))),
        "mean_rgb": [float(v) for v in rgb.reshape(-1, 3).mean(axis=0)],
        "std_rgb": [float(v) for v in rgb.reshape(-1, 3).std(axis=0)],
        "nonempty_rgb": bool(np.any(rgb != 0) and float(rgb.std()) > 0.5),
    }
    if depth is not None and depth.shape == black.shape:
        finite = np.isfinite(depth)
        finite_ratio = float(finite.mean())
        invalid_ratio = float((~finite).mean())
        black_finite_ratio = float(finite[black].mean()) if black.any() else None
        result.update(
            {
                "finite_depth_ratio": finite_ratio,
                "invalid_depth_ratio": invalid_ratio,
                "black_finite_depth_ratio": black_finite_ratio,
            }
        )
        result["depth"] = {
            "finite_ratio": finite_ratio,
            "invalid_or_infinite_ratio": invalid_ratio,
            "black_pixel_count": int(black.sum()),
            "black_pixels_with_finite_depth_ratio": black_finite_ratio,
            "finite_min": float(depth[finite].min()) if finite.any() else None,
            "finite_max": float(depth[finite].max()) if finite.any() else None,
        }
    else:
        result.update({"finite_depth_ratio": None, "invalid_depth_ratio": None, "black_finite_depth_ratio": None})
        result["depth"] = {"status": "missing_or_shape_mismatch"}
    return result


def save_depth(depth: np.ndarray, npy_path: Path, vis_path: Path) -> None:
    np.save(npy_path, depth.astype(np.float32))
    finite = np.isfinite(depth)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [2.0, 98.0])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        vis[finite] = ((1.0 - norm[finite]) * 255.0).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(vis_path)


def make_contact_sheet(records: list[dict[str, Any]], out_path: Path, title: str) -> None:
    if not records:
        Image.new("RGB", (640, 180), (245, 245, 245)).save(out_path)
        return
    cell_w, cell_h, header = 360, 255, 54
    cols = min(3, len(records))
    rows = math.ceil(len(records) / cols)
    sheet = Image.new("RGB", (cols * cell_w, header + rows * cell_h), (242, 244, 247))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 16), title, fill=(15, 20, 28))
    for index, record in enumerate(records):
        x = (index % cols) * cell_w + 8
        y = header + (index // cols) * cell_h + 8
        # Pillow 11.3 as bundled with the Isaac 4.5 environment can delete the
        # private ``_close_fp`` attribute after decoding a PNG.  Its context
        # manager then raises during ``__exit__`` even though the image decoded
        # successfully.  Fully load/copy the tiny number of contact-sheet
        # inputs and let the short-lived object be collected instead.
        source_image = Image.open(record["rgb"])
        source_image.load()
        thumb = source_image.convert("RGB").copy()
        thumb.thumbnail((344, 194), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        metrics = record["metrics"]
        label = f"{record['variant_id']}\nblack={metrics['pure_black_ratio']:.3f} finite={metrics.get('depth',{}).get('finite_ratio')}"
        draw.multiline_text((x, y + 198), label, fill=(18, 24, 32), spacing=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def bounds_record(stage, path: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = stage.GetPseudoRoot()
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    aligned = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True).ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = aligned.GetMin(), aligned.GetMax()
    low = [safe_float(mn[i]) for i in range(3)]
    high = [safe_float(mx[i]) for i in range(3)]
    if any(v is None for v in low + high):
        return {"status": "invalid", "path": str(prim.GetPath())}
    size = [high[i] - low[i] for i in range(3)]
    return {
        "status": "ok",
        "path": str(prim.GetPath()),
        "min": low,
        "max": high,
        "size": size,
        "center": [(low[i] + high[i]) * 0.5 for i in range(3)],
        "diagonal": math.sqrt(sum(v * v for v in size)),
    }


def stage_audit(stage, source_dir: Path) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    counts = {k: 0 for k in ("prim", "mesh", "material", "shader", "camera", "light", "collider", "material_binding")}
    cameras: list[dict[str, Any]] = []
    asset_attributes: list[dict[str, Any]] = []
    material_samples: list[str] = []
    texture_paths: set[str] = set()
    mdl_paths: set[str] = set()
    for prim in stage.Traverse():
        counts["prim"] += 1
        type_name = prim.GetTypeName() or ""
        if prim.IsA(UsdGeom.Mesh):
            counts["mesh"] += 1
        if type_name == "Material":
            counts["material"] += 1
            if len(material_samples) < 30:
                material_samples.append(str(prim.GetPath()))
        if type_name == "Shader":
            counts["shader"] += 1
        if prim.IsA(UsdGeom.Camera):
            counts["camera"] += 1
            camera = UsdGeom.Camera(prim)
            cameras.append(
                {
                    "path": str(prim.GetPath()),
                    "focal_length": json_default(camera.GetFocalLengthAttr().Get()),
                    "horizontal_aperture": json_default(camera.GetHorizontalApertureAttr().Get()),
                    "vertical_aperture": json_default(camera.GetVerticalApertureAttr().Get()),
                    "clipping_range": json_default(camera.GetClippingRangeAttr().Get()),
                    "world_transform": matrix_rows(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),
                }
            )
        if "Light" in type_name:
            counts["light"] += 1
        try:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                counts["collider"] += 1
        except Exception:
            pass
        try:
            if UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets():
                counts["material_binding"] += 1
        except Exception:
            pass
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray) or not attr.HasAuthoredValueOpinion():
                continue
            try:
                value = attr.Get()
                values = value if isinstance(value, (list, tuple)) else [value]
                for item in values:
                    authored = getattr(item, "path", "")
                    resolved = getattr(item, "resolvedPath", "")
                    if not authored:
                        continue
                    row = {"prim": str(prim.GetPath()), "attribute": attr.GetName(), "authored": authored, "resolved": resolved or None}
                    asset_attributes.append(row)
                    suffix = Path(authored).suffix.lower()
                    if suffix in (".png", ".jpg", ".jpeg", ".hdr", ".exr"):
                        texture_paths.add(authored)
                    if suffix == ".mdl" or "mdl" in attr.GetName().lower():
                        mdl_paths.add(authored)
            except Exception:
                pass
    extension_counts: dict[str, int] = {}
    for path in source_dir.rglob("*"):
        if path.is_file():
            key = path.suffix.lower() or "<none>"
            extension_counts[key] = extension_counts.get(key, 0) + 1
    used_layers = []
    for layer in stage.GetUsedLayers():
        used_layers.append({"identifier": layer.identifier, "real_path": layer.realPath or None, "external_references": sorted(layer.GetExternalReferences())})
    unresolved = [item for item in asset_attributes if not item["resolved"]]
    return {
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "bounds": bounds_record(stage, "/World"),
        "counts": counts,
        "authored_cameras": cameras,
        "material_samples": material_samples,
        "asset_attribute_count": len(asset_attributes),
        "unresolved_asset_attribute_count": len(unresolved),
        "unresolved_asset_attributes": unresolved[:100],
        "texture_asset_paths": sorted(texture_paths),
        "mdl_asset_paths": sorted(mdl_paths),
        "hdr_files": [str(p) for p in sorted(source_dir.rglob("*.hdr"))],
        "extension_counts": dict(sorted(extension_counts.items())),
        "used_layers": used_layers,
        "depth_collision_scope": {
            "mesh_count": counts["mesh"],
            "collider_count": counts["collider"],
            "note": "RGB background pixels with infinite distance are not collision- or material-randomizable 3D geometry.",
        },
    }


def choose_camera(stage, requested: str, bounds: dict[str, Any], up_axis: str, meters_per_unit: float):
    from pxr import Gf, Sdf, Usd, UsdGeom

    fixed_route_views = {
        "fixed_kyoto01": ((-17.0, -7.0, 1.6), (-7.0, -7.0, 1.3)),
        "fixed_kyoto03": ((-7.0, -7.0, 1.6), (3.0, -7.0, 1.3)),
        "fixed_paris": ((-82.5, -4.0, 1.6), (-72.0, -4.0, 1.3)),
    }
    # The Training Scenes use recorded route cameras.  Avoid traversing their very
    # large composed stages when the requested camera is already deterministic.
    source = None
    source_cameras = []
    if requested != "auto" and requested not in fixed_route_views:
        candidate = stage.GetPrimAtPath(requested)
        if candidate and candidate.IsValid() and candidate.IsA(UsdGeom.Camera):
            source = candidate
    if requested not in fixed_route_views and source is None:
        source_cameras = [
            prim
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Camera)
            and not str(prim.GetPath()).startswith("/UrbanVerseEvaluation")
            and not str(prim.GetPath()).startswith("/OmniverseKit_")
        ]
        source_cameras.sort(key=lambda prim: str(prim.GetPath()))
    if source is None and source_cameras:
        source = next((prim for prim in source_cameras if str(prim.GetPath()).endswith("Camera_01")), source_cameras[0])
    camera = UsdGeom.Camera.Define(stage, "/UrbanVerseEvaluation/Camera")
    xform = UsdGeom.Xformable(camera.GetPrim())
    op = xform.AddTransformOp()
    if requested in fixed_route_views:
        eye_values, target_values = fixed_route_views[requested]
        matrix = Gf.Matrix4d(1.0).SetLookAt(Gf.Vec3d(*eye_values), Gf.Vec3d(*target_values), Gf.Vec3d(0, 0, 1)).GetInverse()
        op.Set(matrix)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(15.2908)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
        camera.GetPrim().CreateAttribute("evaluation:fixedRouteCamera", Sdf.ValueTypeNames.Bool).Set(True)
        method = "recorded_fixed_route_camera"
        source_path = None
    elif source is not None:
        src_camera = UsdGeom.Camera(source)
        for name in ("focalLength", "horizontalAperture", "verticalAperture", "horizontalApertureOffset", "verticalApertureOffset", "clippingRange", "fStop", "focusDistance"):
            src_attr = source.GetAttribute(name)
            if src_attr and src_attr.HasValue():
                camera.GetPrim().CreateAttribute(name, src_attr.GetTypeName()).Set(src_attr.Get())
        matrix = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        op.Set(matrix)
        method = "authored_camera_copy"
        source_path = str(source.GetPath())
    else:
        scale = 1.0 / max(meters_per_unit, 1e-9)
        center = bounds.get("center", [0.0, 0.0, 0.0])
        size = bounds.get("size", [10.0 * scale, 10.0 * scale, 4.0 * scale])
        if up_axis.upper() == "Y":
            horizontal = max(8.0 * scale, math.hypot(size[0], size[2]))
            floor = bounds.get("min", [0.0, center[1] - 2.0 * scale, 0.0])[1]
            eye = Gf.Vec3d(center[0] - 0.18 * horizontal, floor + 1.6 * scale, center[2] - 0.20 * horizontal)
            target = Gf.Vec3d(center[0] + 0.10 * horizontal, floor + 1.35 * scale, center[2] + 0.05 * horizontal)
            up = Gf.Vec3d(0, 1, 0)
        else:
            horizontal = max(8.0 * scale, math.hypot(size[0], size[1]))
            floor = bounds.get("min", [0.0, 0.0, center[2] - 2.0 * scale])[2]
            eye = Gf.Vec3d(center[0] - 0.18 * horizontal, center[1] - 0.20 * horizontal, floor + 1.6 * scale)
            target = Gf.Vec3d(center[0] + 0.10 * horizontal, center[1] + 0.05 * horizontal, floor + 1.35 * scale)
            up = Gf.Vec3d(0, 0, 1)
        matrix = Gf.Matrix4d(1.0).SetLookAt(eye, target, up).GetInverse()
        op.Set(matrix)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(15.2908)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01 * scale, 10000.0 * scale))
        camera.GetPrim().CreateAttribute("evaluation:fixedCamera", Sdf.ValueTypeNames.Bool).Set(True)
        method = "bounds_derived_fixed_camera"
        source_path = None
    metadata = {
        "prim_path": str(camera.GetPath()),
        "selection_method": method,
        "source_camera_path": source_path,
        "requested_camera_path": requested,
        "source_metadata_note": (
            "Training route coordinates follow the established Z-up route pipeline even though source stage metadata says Y-up/0.01; "
            "the fixed matrix and rendered depth are the acceptance evidence."
            if requested in fixed_route_views
            else None
        ),
        "world_transform": matrix_rows(matrix),
        "focal_length": json_default(camera.GetFocalLengthAttr().Get()),
        "horizontal_aperture": json_default(camera.GetHorizontalApertureAttr().Get()),
        "vertical_aperture": json_default(camera.GetVerticalApertureAttr().Get()),
        "clipping_range": json_default(camera.GetClippingRangeAttr().Get()),
    }
    return camera, op, matrix, metadata


def define_preview_material(stage, path: str, color: list[float], metallic: float, roughness: float):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def mesh_candidates(stage) -> tuple[list[Any], list[Any]]:
    from pxr import Usd, UsdGeom

    semantic_ground: list[tuple[float, Any]] = []
    geometric_ground: list[tuple[float, Any]] = []
    objects = []
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        lower = path.lower()
        if any(token in lower for token in ("collision", "collider", "physics")):
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            size = [abs(float(box.GetMax()[i] - box.GetMin()[i])) for i in range(3)]
        except Exception:
            size = [0.0, 0.0, 0.0]
        if "street_light" in lower or "streetlight" in lower:
            objects.append(prim)
            continue
        # Route capture mounts the complete source stage below
        # /World/ground/terrain.  Treating those wrapper ancestors as semantic
        # hints classifies every building and vehicle as ground.  Restrict the
        # hint to the local tail and to actual surface-category names; the
        # geometric flatness test remains the fallback for unnamed road meshes.
        local_tail = "/".join(lower.strip("/").split("/")[-6:])
        explicit_object = any(
            token in local_tail
            for token in (
                "vehicle_",
                "street_cabinet",
                "bus_shelter",
                "building_",
                "window",
                "door",
                "wall",
                "roof",
                "tree",
                "vegetation",
                "street_light",
                "streetlight",
                "traffic_light",
                "lamp",
                "pole",
                "fence",
                "barrier",
                "bench",
                "trash",
                "garbage",
                "hydrant",
                "traffic_sign",
                "crosswalk",
                "whiteline",
                "yellowline",
                "road_marking",
                "roadmarking",
            )
        )
        if explicit_object:
            objects.append(prim)
            continue
        keyword = any(
            token in local_tail
            for token in (
                "road",
                "street",
                "walkable",
                "sidewalk",
                "floor",
                "plaza",
                "pavement",
                "lane",
                "nearbuffer",
                "groundplane",
                "terrain",
            )
        )
        # A generic ``min(size)`` test also accepts thin vertical walls.  This
        # pipeline uses Z as its practical vertical axis, so only broad XY
        # meshes that are thin in Z are valid unnamed ground candidates.
        horizontal_extent = max(size[0], size[1])
        flat = horizontal_extent > 3.0 and size[2] < horizontal_extent * 0.08
        area = max(size[0] * size[1], 0.0)
        if keyword:
            semantic_ground.append((area, prim))
        elif flat:
            geometric_ground.append((area, prim))
        else:
            objects.append(prim)
    # Traverse order is an asset-authoring accident.  A fixed first-N limit
    # previously covered only one city block and left the road visible from
    # another block unchanged.  Bind named surface meshes first and sort every
    # group by footprint so the recorded selection is deterministic and favors
    # the broad, visible surfaces.  Geometric fallbacks remain last because a
    # flat vehicle panel or cabinet part is not semantically safe to relabel.
    semantic_ground.sort(key=lambda item: (-item[0], str(item[1].GetPath())))
    geometric_ground.sort(key=lambda item: (-item[0], str(item[1].GetPath())))
    ground = [prim for _, prim in semantic_ground] + [prim for _, prim in geometric_ground]
    return ground, objects


def bind_targets(targets: list[Any], material, limit: int) -> dict[str, Any]:
    from pxr import UsdShade

    bound, failed, candidate_to_binding_owner = [], [], []
    seen_owners: set[str] = set()
    for prim in targets:
        if len(bound) >= limit:
            break
        try:
            # CraftBench authors visual material bindings on ancestors such as
            # ``Lane/Lane`` with ``strongerThanDescendants``.  Binding a child
            # mesh therefore writes a valid relationship that is nevertheless
            # ignored by USD's binding resolver.  Override the prim that owns
            # the currently resolved visual relationship in the session layer;
            # authoring the same relationship path in a stronger layer changes
            # appearance without touching the source or its physics material.
            _, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            owner = relationship.GetPrim() if relationship else prim
            owner_path = str(owner.GetPath())
            if owner_path in seen_owners:
                continue
            seen_owners.add(owner_path)
            UsdShade.MaterialBindingAPI.Apply(owner).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            resolved_material, resolved_relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            resolved_path = str(resolved_material.GetPath()) if resolved_material else None
            expected_path = str(material.GetPath())
            if resolved_path != expected_path:
                failed.append(
                    {
                        "candidate_path": str(prim.GetPath()),
                        "binding_owner_path": owner_path,
                        "error": f"material readback mismatch: expected {expected_path}, got {resolved_path}",
                    }
                )
                continue
            bound.append(owner_path)
            candidate_to_binding_owner.append(
                {
                    "candidate_path": str(prim.GetPath()),
                    "binding_owner_path": owner_path,
                    "resolved_relationship": str(resolved_relationship.GetPath())
                    if resolved_relationship
                    else None,
                    "resolved_material": resolved_path,
                }
            )
        except Exception as exc:
            failed.append({"path": str(prim.GetPath()), "error": str(exc)})
    return {
        "target_count": len(bound),
        "target_paths": bound,
        "candidate_to_binding_owner": candidate_to_binding_owner,
        "resolved_material_readback_all_match": len(failed) == 0,
        "failed": failed[:20],
    }


def apply_variant(
    stage,
    variant: dict[str, Any],
    camera,
    camera_op,
    base_matrix,
    base_focal: float,
    base_horizontal_aperture: float,
    base_vertical_aperture: float,
    meters_per_unit: float,
    seed: int,
    reference_depth: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
):
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    rng = np.random.default_rng(seed + int(variant["seed_offset"]))
    vid = variant["id"]
    kind = variant.get("kind", vid)
    profile = variant.get("profile")
    record: dict[str, Any] = {
        "id": vid,
        "kind": kind,
        "profile": profile,
        "tests": variant["tests"],
        "seed": seed + int(variant["seed_offset"]),
        "proxy": False,
        "display_name_zh": variant.get("display_name_zh", vid),
        "explanation_zh": variant.get("explanation_zh", ""),
        "strength_note_zh": "增强可视化档位：用于清楚展示差异，不等于推荐训练默认范围。" if kind != "baseline" else "原始外观对照。",
    }
    camera_op.Set(base_matrix)
    camera.GetFocalLengthAttr().Set(base_focal)
    camera.GetHorizontalApertureAttr().Set(base_horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(base_vertical_aperture)
    if kind == "baseline":
        record["application"] = "none; source appearance"
    elif kind == "light":
        import carb

        dome = UsdLux.DomeLight.Define(stage, "/UrbanVerseEvaluation/Variant/Dome")
        distant = UsdLux.DistantLight.Define(stage, "/UrbanVerseEvaluation/Variant/Sun")
        if profile == "bright_noon":
            intensity = float(rng.uniform(900.0, 1150.0))
            sun_intensity = float(rng.uniform(9000.0, 12500.0))
            dome_color = [0.82, 0.91, 1.0]
            sun_color = [1.0, 0.97, 0.90]
            angles = [float(rng.uniform(-82, -68)), float(rng.uniform(-18, 18)), float(rng.uniform(-8, 8))]
            exposure = 0.25
            time_system = "bright cool-white noon lighting proxy"
        else:
            intensity = float(rng.uniform(90.0, 180.0))
            sun_intensity = float(rng.uniform(1800.0, 3200.0))
            dome_color = [0.34, 0.20, 0.42]
            sun_color = [1.0, 0.28, 0.055]
            angles = [float(rng.uniform(-16, -6)), float(rng.uniform(48, 75)), float(rng.uniform(-15, 15))]
            exposure = -0.55
            time_system = "low-angle warm sunset lighting proxy"
        dome.CreateIntensityAttr(intensity)
        dome.CreateColorAttr(Gf.Vec3f(*dome_color))
        distant.CreateIntensityAttr(sun_intensity)
        distant.CreateColorAttr(Gf.Vec3f(*sun_color))
        rotate = UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp()
        rotate.Set(Gf.Vec3f(*angles))
        carb.settings.get_settings().set("/rtx/post/tonemap/exposure", exposure)
        record.update({"dome_intensity": intensity, "dome_color": dome_color, "distant_intensity": sun_intensity, "distant_color": sun_color, "distant_rotation_xyz_deg": angles, "exposure_ev": exposure, "time_system": time_system})
    elif kind == "weather":
        import carb

        dome = UsdLux.DomeLight.Define(stage, "/UrbanVerseEvaluation/Variant/WeatherDome")
        if profile == "overcast_cool":
            intensity = float(rng.uniform(1000.0, 1350.0))
            color = [0.48, 0.68, 1.0]
            exposure = -0.15
            weather = "strong cool overcast sky/light proxy only"
        else:
            intensity = float(rng.uniform(100.0, 180.0))
            color = [0.075, 0.12, 0.36]
            exposure = -1.20
            weather = "dark blue pre-storm sky/light proxy only"
        dome.CreateIntensityAttr(intensity)
        dome.CreateColorAttr(Gf.Vec3f(*color))
        carb.settings.get_settings().set("/rtx/post/tonemap/exposure", exposure)
        record.update({"proxy": True, "weather": weather, "dome_intensity": intensity, "dome_color": color, "exposure_ev": exposure, "not_implemented": ["rain", "snow", "fog volume", "wet ground", "puddles", "weather interaction"]})
    elif kind == "road_overlay":
        from pxr import Gf, UsdShade

        if reference_depth is None or image_size is None:
            raise RuntimeError("road overlay requires baseline distance-to-camera and image size")
        height, width = reference_depth.shape
        target_u = width // 2
        target_v = int(height * 0.78)
        depth_m = float(reference_depth[target_v, target_u])
        if not math.isfinite(depth_m) or depth_m <= 0:
            crop = reference_depth[int(height * 0.68) : int(height * 0.88), int(width * 0.44) : int(width * 0.56)]
            finite = crop[np.isfinite(crop) & (crop > 0)]
            if not finite.size:
                raise RuntimeError("no finite road-region depth available for overlay placement")
            depth_m = float(np.median(finite))

        focal = max(float(base_focal), 1e-6)
        x_camera = ((target_u + 0.5) / width - 0.5) * float(base_horizontal_aperture) / focal
        y_camera = (0.5 - (target_v + 0.5) / height) * float(base_vertical_aperture) / focal
        ray_world = base_matrix.TransformDir(Gf.Vec3d(x_camera, y_camera, -1.0)).GetNormalized()
        eye = base_matrix.ExtractTranslation()
        distance_stage_units = depth_m / max(float(meters_per_unit), 1e-9)
        road_point = eye + ray_world * distance_stage_units
        forward = base_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
        flat_forward = Gf.Vec3d(forward[0], forward[1], 0.0).GetNormalized()
        overlay_width = 8.0
        overlay_length = 44.0
        overlay_thickness = 0.012
        overlay_center = road_point + flat_forward * 14.0 + Gf.Vec3d(0.0, 0.0, overlay_thickness * 0.65)
        yaw_deg = math.degrees(math.atan2(float(flat_forward[1]), float(flat_forward[0]))) - 90.0

        surface = UsdGeom.Cube.Define(stage, "/UrbanVerseEvaluation/Variant/DarkAsphaltRoadSurface")
        surface.CreateSizeAttr(1.0)
        surface_xform = UsdGeom.Xformable(surface.GetPrim())
        surface_xform.AddTranslateOp().Set(overlay_center)
        surface_xform.AddRotateZOp().Set(yaw_deg)
        surface_xform.AddScaleOp().Set(Gf.Vec3f(overlay_width, overlay_length, overlay_thickness))
        color = [0.055, 0.062, 0.068]
        metallic, roughness = 0.0, 0.88
        material = define_preview_material(
            stage,
            "/UrbanVerseEvaluation/Variant/DarkAsphaltMaterial",
            color,
            metallic,
            roughness,
        )
        UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose=UsdShade.Tokens.allPurpose,
        )
        record.update(
            {
                "proxy": True,
                "surface_path": str(surface.GetPath()),
                "material_path": str(material.GetPath()),
                "diffuse_color": color,
                "metallic": metallic,
                "roughness": roughness,
                "sample_pixel_uv": [target_u, target_v],
                "sample_distance_m": depth_m,
                "sample_distance_stage_units": distance_stage_units,
                "road_point_world": [float(road_point[i]) for i in range(3)],
                "overlay_center_world": [float(overlay_center[i]) for i in range(3)],
                "overlay_size_stage_units": [overlay_width, overlay_length, overlay_thickness],
                "yaw_deg": yaw_deg,
                "limitation": "run-local road-surface overlay proxy; it does not repair or replace the source USD road material binding",
            }
        )
    elif kind == "material":
        target_kind = str(variant["target_kind"])
        if profile == "warm_rough":
            color = [float(rng.uniform(0.82, 0.96)), float(rng.uniform(0.035, 0.11)), float(rng.uniform(0.01, 0.045))]
            metallic, roughness = 0.0, 0.95
        elif profile == "cool_glossy":
            color = [float(rng.uniform(0.01, 0.06)), float(rng.uniform(0.34, 0.56)), float(rng.uniform(0.78, 0.96))]
            metallic, roughness = 0.20, 0.22
        elif profile == "neutral_preview_compat":
            color = [0.28, 0.30, 0.31]
            metallic, roughness = 0.0, 0.82
        elif profile == "dark_asphalt_replacement":
            color = [0.075, 0.082, 0.088]
            metallic, roughness = 0.0, 0.93
        elif profile == "magenta":
            color = [float(rng.uniform(0.82, 0.96)), float(rng.uniform(0.005, 0.04)), float(rng.uniform(0.54, 0.82))]
            metallic, roughness = 0.0, 0.58
        else:
            color = [float(rng.uniform(0.005, 0.04)), float(rng.uniform(0.68, 0.90)), float(rng.uniform(0.82, 0.98))]
            metallic, roughness = 0.0, 0.42
        material = define_preview_material(stage, "/UrbanVerseEvaluation/Variant/RandomizedMaterial", color, metallic, roughness)
        ground, objects = mesh_candidates(stage)
        # CraftBench roads are split across many repeated city-block payloads.
        # 256 keeps the override bounded while covering all named surface
        # meshes in the representative scenes.  The exact prim list is written
        # to metadata and source USDs remain untouched.
        binding = bind_targets(ground if target_kind == "ground" else objects, material, limit=256 if target_kind == "ground" else 32)
        record.update(
            {
                "target_kind": target_kind,
                "material_path": str(material.GetPath()),
                "diffuse_color": color,
                "metallic": metallic,
                "roughness": roughness,
                "binding": binding,
                "proxy": binding["target_count"] == 0,
                "ground_treatment_role": (
                    "non-destructive MDL/projection compatibility fallback"
                    if profile == "neutral_preview_compat"
                    else "non-destructive training appearance replacement"
                    if profile == "dark_asphalt_replacement"
                    else "appearance randomization"
                ),
                "geometry_depth_collision_unchanged_by_design": True,
            }
        )
    elif kind == "camera":
        # UrbanVerse geometry and the established navigation routes use world
        # coordinates as practical metres even though these source roots carry
        # inconsistent 0.01 metersPerUnit metadata.  Scale=1 is validated by
        # the fixed cameras and avoids moving a camera tens of world units.
        scale = 1.0
        if profile == "wide_bright":
            translation_m = [float(v) for v in rng.uniform([0.24, -0.32, 0.10], [0.42, -0.16, 0.22])]
            yaw_deg = float(rng.uniform(-9.0, -6.0))
            focal_scale = float(rng.uniform(0.58, 0.70))
            aperture_scale = float(rng.uniform(1.12, 1.24))
            exposure = float(rng.uniform(1.05, 1.35))
        else:
            translation_m = [float(v) for v in rng.uniform([-0.40, 0.14, -0.12], [-0.22, 0.30, -0.03])]
            yaw_deg = float(rng.uniform(6.0, 9.0))
            focal_scale = float(rng.uniform(1.38, 1.58))
            aperture_scale = float(rng.uniform(0.76, 0.88))
            exposure = float(rng.uniform(-1.35, -1.05))
        matrix = Gf.Matrix4d(base_matrix)
        translation = matrix.ExtractTranslation()
        translation += Gf.Vec3d(*(v * scale for v in translation_m))
        matrix.SetTranslateOnly(translation)
        up = Gf.Vec3d(0, 0, 1)
        matrix = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(up, yaw_deg)) * matrix
        camera_op.Set(matrix)
        camera.GetFocalLengthAttr().Set(base_focal * focal_scale)
        camera.GetHorizontalApertureAttr().Set(base_horizontal_aperture * aperture_scale)
        camera.GetVerticalApertureAttr().Set(base_vertical_aperture * aperture_scale)
        import carb

        carb.settings.get_settings().set("/rtx/post/tonemap/exposure", exposure)
        record.update({"translation_noise_m": translation_m, "rotation_noise_yaw_deg": yaw_deg, "focal_length_scale": focal_scale, "actual_focal_length": base_focal * focal_scale, "aperture_scale": aperture_scale, "actual_horizontal_aperture": base_horizontal_aperture * aperture_scale, "actual_vertical_aperture": base_vertical_aperture * aperture_scale, "exposure_ev": exposure, "world_transform": matrix_rows(matrix)})
    elif kind == "reflection":
        scale = 1.0
        forward = base_matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
        eye = base_matrix.ExtractTranslation()
        sphere = UsdGeom.Sphere.Define(stage, "/UrbanVerseEvaluation/Variant/ReflectiveProbe")
        if profile == "mirror":
            radius, distance, color, roughness = 0.82, 4.0, [0.82, 0.88, 0.96], 0.008
        else:
            radius, distance, color, roughness = 0.68, 3.6, [0.95, 0.48, 0.055], 0.52
        sphere.CreateRadiusAttr(radius * scale)
        sphere.AddTranslateOp().Set(eye + forward * (distance * scale))
        material = define_preview_material(stage, "/UrbanVerseEvaluation/Variant/ReflectiveMaterial", color, 1.0, roughness)
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
        record.update({"proxy": True, "probe_path": str(sphere.GetPath()), "material_path": str(material.GetPath()), "radius_m": radius, "distance_m": distance, "diffuse_color": color, "metallic": 1.0, "roughness": roughness, "limitation": "renderer capability proxy; not an exhaustive audit of native asset reflections"})
    return record


def annotator_data(annotator) -> np.ndarray:
    value = annotator.get_data()
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    return np.asarray(value)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    captures = run_dir / "captures"
    metadata = run_dir / "metadata"
    visualizations = run_dir / "visualizations"
    wrapper_dir = run_dir / "wrapper"
    for path in (captures, metadata, visualizations, wrapper_dir):
        path.mkdir(parents=True, exist_ok=True)
    source_usd = args.usd.resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(source_usd)
    wrapper_path = wrapper_dir / f"{args.scene}_wrapper.usda"
    template = args.wrapper_template.read_text(encoding="utf-8")
    wrapper_path.write_text(template.replace("SOURCE_USD", str(source_usd)), encoding="utf-8")
    summary_path = metadata / "summary.json"
    summary: dict[str, Any] = {
        "status": "failed",
        "mode": args.mode,
        "family": args.family,
        "scene": args.scene,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_usd": str(source_usd),
        "wrapper_usd": str(wrapper_path),
    }
    write_json(summary_path, summary)
    simulation_app = None
    try:
        from isaacsim import SimulationApp

        launch = {
            "headless": True,
            "renderer": "RayTracedLighting",
            "width": args.width,
            "height": args.height,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "multi_gpu": False,
        }
        print("SIMULATION_APP " + json.dumps(launch, sort_keys=True), flush=True)
        simulation_app = SimulationApp(launch, experience=str(args.experience.resolve()))
        import carb
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        settings = carb.settings.get_settings()
        settings.set("/rtx/multiGpu/enabled", False)
        settings.set("/physics/cudaDevice", args.gpu)
        settings.set("/renderer/activeGpu", args.gpu)
        settings.set("/rtx/post/tonemap/exposure", 0.0)
        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(wrapper_path)):
            raise RuntimeError("open_stage returned false")
        last_loading = None
        for update in range(2400):
            simulation_app.update()
            try:
                last_loading = tuple(ctx.get_stage_loading_status())
                if not last_loading or (len(last_loading) >= 3 and (int(last_loading[-1]) == 0 or int(last_loading[-2]) >= int(last_loading[-1]))):
                    break
            except Exception:
                if update > 10:
                    break
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError("stage is None")
        audit = stage_audit(stage, source_usd.parent)
        write_json(metadata / "asset_audit.json", audit)
        root = stage.GetRootLayer()
        stage.SetEditTarget(root)
        stage.DefinePrim("/UrbanVerseEvaluation", "Scope")
        camera, camera_op, base_matrix, camera_metadata = choose_camera(
            stage, args.camera_path, audit["bounds"], audit["up_axis"], audit["meters_per_unit"]
        )
        root.Save()
        write_json(metadata / "camera.json", camera_metadata)
        git_commit = run_text(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"])
        gpu_query = run_text(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=index,name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"])
        kit_version = None
        try:
            kit_version = omni.kit.app.get_app().get_kit_version()
        except Exception:
            kit_version = settings.get("/app/kitVersion") or settings.get("/app/buildVersion")
        environment = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": platform.node(),
            "gpu_index": args.gpu,
            "gpu_name_driver_utilization": gpu_query,
            "driver": run_text(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=driver_version", "--format=csv,noheader"]),
            "isaac_sim_version": package_version("isaacsim"),
            "replicator_version": package_version("isaacsim-replicator"),
            "kit_version": json_default(kit_version),
            "python_version": sys.version.replace("\n", " "),
            "git_commit": git_commit,
            "command_line": [sys.executable, *sys.argv],
            "renderer": "RayTracedLighting",
            "resolution": [args.width, args.height],
            "camera": camera_metadata,
            "source_usd": str(source_usd),
            "source_usd_sha256": sha256(source_usd),
            "source_tar": str(args.tar.resolve()) if args.tar else None,
            "source_tar_sha256": sha256(args.tar.resolve()) if args.tar else None,
            "wrapper_path": str(wrapper_path),
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "random_seed": args.seed,
            "settle_frames": args.settle_frames,
            "rt_subframes": args.rt_subframes,
            "settings_readback": {
                "renderer_active_gpu": settings.get("/renderer/activeGpu"),
                "physics_cuda_device": settings.get("/physics/cudaDevice"),
                "multi_gpu": settings.get("/rtx/multiGpu/enabled"),
            },
        }
        write_json(metadata / "environment.json", environment)
        if args.mode == "audit":
            Image.new("RGB", (640, 160), (242, 244, 247)).save(visualizations / "contact_sheet.png")
            summary.update({"status": "success", "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"), "audit": audit, "camera": camera_metadata, "environment": environment})
            write_json(summary_path, summary)
            return 0

        render_product = rep.create.render_product(camera.GetPath(), (args.width, args.height), name="UrbanVerseEvaluationRender")
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        camera_annotator = rep.AnnotatorRegistry.get_annotator("camera_params")
        for annotator in (rgb_annotator, depth_annotator, camera_annotator):
            annotator.attach([render_product])
        if args.mode == "appearance":
            # The first Replicator capture after graph construction is
            # systematically darker than later captures on this 4.5 build.
            # Discard one source-appearance capture so every recorded variant
            # is compared after the same graph warm-up boundary.
            for _ in range(args.settle_frames):
                simulation_app.update()
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
            try:
                rep.orchestrator.wait_until_complete()
            except Exception as exc:
                print(f"WARMUP_WAIT_NOTE {exc!r}", flush=True)
            print("DISCARDED_GRAPH_WARMUP_CAPTURE", flush=True)
        base_focal = float(camera.GetFocalLengthAttr().Get() or 18.0)
        base_horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get() or 20.955)
        base_vertical_aperture = float(camera.GetVerticalApertureAttr().Get() or 15.2908)
        if args.mode == "appearance" and args.variant_filter != "all":
            requested_variants = {value.strip() for value in args.variant_filter.split(",") if value.strip()}
            requested_variants.add("baseline")
            variants = [variant for variant in VARIANTS if variant["id"] in requested_variants]
            missing_variants = sorted(requested_variants - {variant["id"] for variant in variants})
            if missing_variants:
                raise ValueError(f"unknown appearance variants: {missing_variants}")
        else:
            variants = VARIANTS if args.mode == "appearance" else VARIANTS[:1]
        records: list[dict[str, Any]] = []
        baseline_rgb: np.ndarray | None = None
        baseline_depth: np.ndarray | None = None
        session = stage.GetSessionLayer()
        # Replicator authors its render product and graph in the session layer.
        # Keep that layer intact and clear only this dedicated weaker sublayer.
        variant_layer = Sdf.Layer.CreateAnonymous("urbanverse_evaluation_variant.usda")
        session.subLayerPaths.append(variant_layer.identifier)
        for variant in variants:
            variant_layer.Clear()
            stage.SetEditTarget(variant_layer)
            prior_variant_prim_present_after_clear = bool(
                stage.GetPrimAtPath("/UrbanVerseEvaluation/Variant")
            )
            settings.set("/rtx/post/tonemap/exposure", 0.0)
            variant_record = apply_variant(
                stage,
                variant,
                camera,
                camera_op,
                base_matrix,
                base_focal,
                base_horizontal_aperture,
                base_vertical_aperture,
                audit["meters_per_unit"],
                args.seed,
                baseline_depth,
                (args.width, args.height),
            )
            collision_count = sum(
                1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
            )
            variant_record["collision_prim_count_readback"] = collision_count
            variant_record["collision_prim_count_unchanged"] = (
                collision_count == int(audit["counts"]["collider"])
            )
            variant_record["prior_variant_prim_present_after_clear"] = prior_variant_prim_present_after_clear
            for _ in range(args.settle_frames if variant["id"] == "baseline" else max(6, args.settle_frames // 2)):
                simulation_app.update()
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
            try:
                rep.orchestrator.wait_until_complete()
            except Exception as exc:
                print(f"WAIT_NOTE {exc!r}", flush=True)
            rgb = annotator_data(rgb_annotator)
            depth = annotator_data(depth_annotator).squeeze().astype(np.float32)
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                raise RuntimeError(f"unexpected RGB shape {rgb.shape}")
            rgb = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
            variant_dir = captures / variant["id"]
            variant_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = variant_dir / "rgb.png"
            Image.fromarray(rgb, mode="RGB").save(rgb_path)
            save_depth(depth, variant_dir / "distance_to_camera.npy", variant_dir / "distance_to_camera_vis.png")
            camera_params = camera_annotator.get_data()
            write_json(variant_dir / "camera_params.json", camera_params)
            metrics = image_metrics(rgb, depth)
            if variant["id"] == "baseline":
                baseline_rgb = rgb.copy()
                baseline_depth = depth.copy()
            elif variant["id"] == "road_surface_dark_asphalt_overlay_proxy" and baseline_rgb is not None:
                y0, y1 = int(args.height * 0.58), int(args.height * 0.94)
                x0, x1 = int(args.width * 0.25), int(args.width * 0.75)
                base_roi = baseline_rgb[y0:y1, x0:x1].astype(np.float32)
                variant_roi = rgb[y0:y1, x0:x1].astype(np.float32)
                metrics["road_roi"] = {
                    "xyxy": [x0, y0, x1, y1],
                    "baseline_mean_luma": float((0.2126 * base_roi[..., 0] + 0.7152 * base_roi[..., 1] + 0.0722 * base_roi[..., 2]).mean()),
                    "variant_mean_luma": float((0.2126 * variant_roi[..., 0] + 0.7152 * variant_roi[..., 1] + 0.0722 * variant_roi[..., 2]).mean()),
                    "mean_absolute_rgb_difference_255": float(np.abs(base_roi - variant_roi).mean()),
                    "changed_pixel_ratio_gt_12": float(np.any(np.abs(base_roi - variant_roi) > 12.0, axis=2).mean()),
                }
            record = {
                "variant_id": variant["id"],
                "rgb": str(rgb_path),
                "distance": str(variant_dir / "distance_to_camera.npy"),
                "distance_visualization": str(variant_dir / "distance_to_camera_vis.png"),
                "camera_params": str(variant_dir / "camera_params.json"),
                "metrics": metrics,
                "application": variant_record,
            }
            records.append(record)
            print("CAPTURE " + json.dumps({"variant": variant["id"], "metrics": metrics}, default=json_default), flush=True)
        for annotator in (rgb_annotator, depth_annotator, camera_annotator):
            try:
                annotator.detach()
            except Exception:
                pass
        try:
            render_product.destroy()
        except Exception:
            pass
        write_json(metadata / "records.json", {"records": records})
        contact_sheet = visualizations / "contact_sheet.png"
        make_contact_sheet(records, contact_sheet, f"RTX 3090 + Isaac 4.5 | {args.scene} | {args.mode}")
        summary.update(
            {
                "status": "success",
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_s": round(time.perf_counter() - started, 3),
                "audit": audit,
                "camera": camera_metadata,
                "environment": environment,
                "records": records,
                "visualizations": {"contact_sheet": str(contact_sheet)},
            }
        )
        write_json(summary_path, summary)
        return 0
    except Exception as exc:
        traceback.print_exc()
        summary.update({"status": "failed", "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"), "duration_s": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        write_json(summary_path, summary)
        return 1
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
