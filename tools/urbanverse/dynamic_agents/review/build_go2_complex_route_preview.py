#!/usr/bin/env python3
"""Build a CPU-only bird's-eye preview for a multi-turn Go2 reference route.

The preview consumes an existing PhysX global-route raster cache.  It does not
launch Isaac Sim, create a robot, render cameras, or claim dense PhysX route
validation.  Its JSON output is intended to be reviewed before a later capture
run revalidates the accepted route against the live stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from urbanverse.dynamic_agents.navigation import global_route_planner as planner  # noqa: E402


DEFAULT_CONTROL_POINTS = [
    [-60.0, 45.0],
    [-35.0, 45.0],
    [-35.0, 62.0],
    [-20.0, 77.0],
    [5.0, 77.0],
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()
    except Exception:
        return None


def parse_points(value: str | None) -> np.ndarray:
    if not value:
        return np.asarray(DEFAULT_CONTROL_POINTS, dtype=np.float64)
    rows = []
    for token in value.split(";"):
        fields = [float(item.strip()) for item in token.split(",")]
        if len(fields) != 2:
            raise ValueError("each control point must be X,Y")
        rows.append(fields)
    if len(rows) < 2:
        raise ValueError("at least two control points are required")
    return np.asarray(rows, dtype=np.float64)


def signed_turn_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    h1 = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
    h2 = math.atan2(float(c[1] - b[1]), float(c[0] - b[0]))
    return math.degrees((h2 - h1 + math.pi) % (2.0 * math.pi) - math.pi)


def sample_line(a: np.ndarray, b: np.ndarray, spacing: float) -> np.ndarray:
    length = float(np.linalg.norm(b - a))
    count = max(2, int(math.ceil(length / spacing)) + 1)
    return np.linspace(a, b, count)


def rounded_polyline(points: np.ndarray, radius: float, spacing: float = 0.08) -> np.ndarray:
    """Round corners with quadratic Bezier fillets while preserving endpoints."""
    if len(points) == 2:
        return sample_line(points[0], points[1], spacing)
    entries: list[np.ndarray] = []
    exits: list[np.ndarray] = []
    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1 : index + 2]
        incoming = corner - previous
        outgoing = following - corner
        len_in = float(np.linalg.norm(incoming))
        len_out = float(np.linalg.norm(outgoing))
        if len_in < 1e-6 or len_out < 1e-6:
            raise ValueError("control points must be distinct")
        incoming /= len_in
        outgoing /= len_out
        turn = math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))
        tangent = radius * math.tan(turn / 2.0)
        tangent = min(tangent, 0.35 * len_in, 0.35 * len_out)
        entries.append(corner - incoming * tangent)
        exits.append(corner + outgoing * tangent)

    chunks: list[np.ndarray] = []
    cursor = points[0]
    for index, (entry, exit_) in enumerate(zip(entries, exits), start=1):
        chunks.append(sample_line(cursor, entry, spacing)[:-1])
        corner = points[index]
        estimate = float(np.linalg.norm(entry - corner) + np.linalg.norm(exit_ - corner))
        count = max(8, int(math.ceil(estimate / spacing)) + 1)
        t = np.linspace(0.0, 1.0, count)[:, None]
        curve = (1.0 - t) ** 2 * entry + 2.0 * (1.0 - t) * t * corner + t**2 * exit_
        chunks.append(curve[:-1])
        cursor = exit_
    chunks.append(sample_line(cursor, points[-1], spacing))
    return np.vstack(chunks)


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(delta)])
    samples = np.arange(0.0, float(arc[-1]), spacing)
    if not len(samples) or samples[-1] < arc[-1]:
        samples = np.append(samples, arc[-1])
    return np.column_stack([np.interp(samples, arc, points[:, axis]) for axis in range(2)])


def route_metrics(route: np.ndarray, control_points: np.ndarray) -> dict[str, Any]:
    segments = np.diff(route, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    heading_delta = np.diff(headings)
    ds = np.maximum((segment_lengths[:-1] + segment_lengths[1:]) / 2.0, 1e-9)
    curvature = np.abs(heading_delta) / ds
    turns = [
        {
            "control_point_index": index,
            "world_xy": control_points[index].tolist(),
            "signed_turn_deg": round(signed_turn_deg(*control_points[index - 1 : index + 2]), 1),
        }
        for index in range(1, len(control_points) - 1)
    ]
    return {
        "route_length_stage_units": round(float(segment_lengths.sum()), 3),
        "waypoint_count": int(len(route)),
        "waypoint_spacing_stage_units": round(float(np.median(segment_lengths)), 3),
        "max_sampled_curvature_rad_per_stage_unit": round(float(curvature.max()) if len(curvature) else 0.0, 4),
        "cumulative_abs_turn_deg_from_control_polyline": round(sum(abs(row["signed_turn_deg"]) for row in turns), 1),
        "turns": turns,
    }


def load_grid(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def validate_grid_route(route: np.ndarray, grid: dict[str, Any], inflation_radius: float) -> dict[str, Any]:
    traversable = planner.build_traversability(grid)
    inflation_source = traversable | ~grid["allowed"]
    valid = planner.inflate_obstacles(inflation_source, inflation_radius, float(grid["cell_size"])) & grid["allowed"]
    clearance = distance_transform_edt(traversable) * float(grid["cell_size"])
    samples = resample(route, 0.08)
    failed = []
    clearances = []
    for index, (x, y) in enumerate(samples):
        j, i = planner.cell_at(float(x), float(y), grid["xs"], grid["ys"])
        if not bool(valid[j, i]):
            failed.append(index)
        clearances.append(float(clearance[j, i]))
    return {
        "accepted_on_cached_inflated_grid": not failed,
        "sample_spacing_stage_units": 0.08,
        "sample_count": len(samples),
        "failed_sample_indices": failed[:50],
        "inflation_radius_stage_units": inflation_radius,
        "minimum_cached_grid_clearance_stage_units": round(min(clearances), 3),
        "live_dense_physx_validation": "not_run",
    }


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render_preview(
    output: Path,
    grid: dict[str, Any],
    route: np.ndarray,
    control_points: np.ndarray,
    metrics: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    width, height = 1600, 1050
    panel_width = 430
    margin = 70
    map_box = (margin, 110, width - panel_width - 35, height - 70)
    image = Image.new("RGB", (width, height), (20, 27, 33))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = font(36), font(24), font(19)
    draw.text((margin, 30), "Go2 复杂长路线 · 鸟瞰复核候选", fill=(244, 247, 249), font=title_font)

    traversable = planner.build_traversability(grid)
    inflation_source = traversable | ~grid["allowed"]
    valid = planner.inflate_obstacles(inflation_source, 0.55, float(grid["cell_size"])) & grid["allowed"]
    raster = np.zeros(valid.shape + (3,), dtype=np.uint8)
    raster[:] = (132, 140, 145)
    raster[grid["allowed"]] = (211, 216, 208)
    raster[traversable & grid["allowed"]] = (235, 238, 228)
    raster[(grid["obstacle"] | grid["uncertain"]) & grid["allowed"]] = (55, 63, 70)
    raster[valid] = (224, 232, 218)
    map_img = Image.fromarray(np.flipud(raster), mode="RGB").resize(
        (map_box[2] - map_box[0], map_box[3] - map_box[1]), Image.Resampling.NEAREST
    )
    image.paste(map_img, (map_box[0], map_box[1]))
    draw.rectangle(map_box, outline=(235, 239, 242), width=3)

    x0, x1, y0, y1 = [float(v) for v in grid["bounds_su"]]

    def xy(point: np.ndarray | list[float]) -> tuple[int, int]:
        x = map_box[0] + (float(point[0]) - x0) / (x1 - x0) * (map_box[2] - map_box[0])
        y = map_box[3] - (float(point[1]) - y0) / (y1 - y0) * (map_box[3] - map_box[1])
        return int(round(x)), int(round(y))

    pixels = [xy(point) for point in route]
    draw.line(pixels, fill=(0, 24, 32), width=12, joint="curve")
    draw.line(pixels, fill=(20, 205, 255), width=7, joint="curve")
    for index, point in enumerate(control_points):
        px, py = xy(point)
        color = (52, 224, 123) if index == 0 else (255, 84, 93) if index == len(control_points) - 1 else (255, 185, 50)
        draw.ellipse((px - 11, py - 11, px + 11, py + 11), fill=color, outline=(10, 20, 25), width=3)
        label = "S" if index == 0 else "G" if index == len(control_points) - 1 else f"T{index}"
        draw.text((px + 14, py - 16), label, fill=(12, 18, 22), stroke_width=3, stroke_fill=(245, 247, 242), font=small_font)

    # 10-unit scale and north marker.
    scale_px = int(round(10.0 / (x1 - x0) * (map_box[2] - map_box[0])))
    sx, sy = map_box[0] + 35, map_box[3] - 35
    draw.line((sx, sy, sx + scale_px, sy), fill=(25, 31, 36), width=7)
    draw.text((sx, sy - 33), "10 场景单位", fill=(25, 31, 36), font=small_font)
    nx, ny = map_box[2] - 45, map_box[1] + 75
    draw.polygon([(nx, ny - 45), (nx - 16, ny), (nx + 16, ny)], fill=(25, 31, 36))
    draw.text((nx - 9, ny + 6), "N", fill=(25, 31, 36), font=body_font)

    px = width - panel_width + 10
    draw.rounded_rectangle((px, 110, width - 35, height - 70), radius=18, fill=(31, 40, 47), outline=(77, 91, 101), width=2)
    y = 140
    lines = [
        ("路线摘要", (245, 247, 249), body_font),
        (f"长度：{metrics['route_length_stage_units']:.1f} 场景单位", (20, 205, 255), body_font),
        (f"参考点：{metrics['waypoint_count']} 个", (224, 230, 234), small_font),
        (f"累计转角：{metrics['cumulative_abs_turn_deg_from_control_polyline']:.0f}°", (224, 230, 234), small_font),
        (f"最大曲率：{metrics['max_sampled_curvature_rad_per_stage_unit']:.3f} rad/u", (224, 230, 234), small_font),
        ("", (0, 0, 0), small_font),
        ("转弯结构", (245, 247, 249), body_font),
    ]
    for row in metrics["turns"]:
        direction = "左" if row["signed_turn_deg"] > 0 else "右"
        lines.append((f"T{row['control_point_index']}：{direction}转 {abs(row['signed_turn_deg']):.0f}°", (255, 190, 62), small_font))
    lines.extend(
        [
            ("", (0, 0, 0), small_font),
            ("缓存级安全检查", (245, 247, 249), body_font),
            ("通过" if validation["accepted_on_cached_inflated_grid"] else "未通过", (52, 224, 123) if validation["accepted_on_cached_inflated_grid"] else (255, 84, 93), body_font),
            (f"最小栅格净空：{validation['minimum_cached_grid_clearance_stage_units']:.2f}", (224, 230, 234), small_font),
            ("", (0, 0, 0), small_font),
            ("图例", (245, 247, 249), body_font),
            ("青色：候选参考轨迹", (20, 205, 255), small_font),
            ("浅色：可通行区域", (224, 232, 218), small_font),
            ("深色：障碍/不确定", (146, 155, 162), small_font),
        ]
    )
    for text_value, color, selected_font in lines:
        draw.text((px + 28, y), text_value, fill=color, font=selected_font)
        y += 43 if selected_font == body_font else 34

    warning = "仅供路线复核：尚未启动 Go2、未采集相机数据、未运行逐段实时 PhysX 密集验证"
    draw.rounded_rectangle((px + 18, height - 205, width - 52, height - 92), radius=12, fill=(92, 62, 20))
    # Manual wrapping keeps CJK text readable with Pillow.
    draw.text((px + 35, height - 188), warning[:21], fill=(255, 232, 187), font=small_font)
    draw.text((px + 35, height - 154), warning[21:42], fill=(255, 232, 187), font=small_font)
    draw.text((px + 35, height - 120), warning[42:], fill=(255, 232, 187), font=small_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", default="Asia_China_Beijing_walk_01_Cousin_29")
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--control-points", help="semicolon-separated X,Y pairs")
    parser.add_argument("--corner-radius", type=float, default=3.0)
    parser.add_argument("--waypoint-spacing", type=float, default=0.4)
    parser.add_argument("--minimum-length", type=float, default=50.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"
    visualization_dir = output_dir / "visualizations"
    captures_dir = output_dir / "captures"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.grid_cache.resolve()
    grid = load_grid(grid_path)
    control_points = parse_points(args.control_points)
    dense = rounded_polyline(control_points, args.corner_radius)
    route = resample(dense, args.waypoint_spacing)
    metrics = route_metrics(route, control_points)
    validation = validate_grid_route(dense, grid, inflation_radius=0.55)
    accepted = (
        validation["accepted_on_cached_inflated_grid"]
        and metrics["route_length_stage_units"] >= args.minimum_length
    )

    route_config = json.loads(args.route_config.read_text(encoding="utf-8"))
    scene = next(row for row in route_config["scenes"] if row["slug"] == args.scene)
    source_usd = (PROJECT_ROOT / scene["usd"]).resolve()
    source_hashes = {str(source_usd): sha256(source_usd)}
    if scene.get("tar"):
        source_tar = (PROJECT_ROOT / scene["tar"]).resolve()
        source_hashes[str(source_tar)] = sha256(source_tar)

    plan = {
        "schema": "urbanverse_go2_complex_route_preview_v1",
        "status": "candidate_accepted_on_cached_grid" if accepted else "candidate_rejected",
        "scene": args.scene,
        "control_points_world_xy": control_points.tolist(),
        "waypoints_world_xy": route.tolist(),
        "corner_radius_stage_units": args.corner_radius,
        "metrics": metrics,
        "validation": validation,
        "provenance": {
            "grid_cache": str(grid_path),
            "grid_cache_sha256": sha256(grid_path),
            "grid_bounds_stage_units": [float(v) for v in grid["bounds_su"]],
            "grid_cell_size_stage_units": float(grid["cell_size"]),
            "source_asset_hashes": source_hashes,
        },
        "limitations": [
            "CPU-only preview generated from an existing PhysX raycast cache.",
            "No Go2 instance, camera capture, renderer, or locomotion policy was run.",
            "Before motion, the exact delivered route must pass live dense PhysX corridor and endpoint validation.",
            "Source coordinates are reported as scene units because source metersPerUnit metadata conflicts with visually validated metre-like route values.",
        ],
    }
    write_json(metadata_dir / "route_plan.json", plan)
    write_json(
        metadata_dir / "environment.json",
        {
            "timestamp": now(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status_short": command_output(["git", "status", "--short"]),
            "command_line": " ".join(sys.argv),
            "execution": "CPU-only cached-grid route preview",
            "physical_gpu_selected": None,
            "renderer": None,
            "isaac_sim_run": False,
            "nvidia_smi": command_output(["nvidia-smi"]),
            "scene": args.scene,
            "source_asset_hashes": source_hashes,
            "grid_cache": str(grid_path),
            "grid_cache_sha256": sha256(grid_path),
        },
    )
    image_path = visualization_dir / "complex_route_birds_eye.png"
    render_preview(image_path, grid, route, control_points, metrics, validation)
    capture_path = captures_dir / "complex_route_birds_eye_rgb.png"
    shutil.copy2(image_path, capture_path)
    write_json(
        metadata_dir / "summary.json",
        {
            "timestamp": now(),
            "status": "success" if accepted else "failed",
            "scene": args.scene,
            "route_length_stage_units": metrics["route_length_stage_units"],
            "turns": metrics["turns"],
            "cached_grid_validation_passed": validation["accepted_on_cached_inflated_grid"],
            "live_dense_physx_validation": "not_run",
            "go2_data_collection": "not_run",
            "visualization": str(image_path),
            "capture_rgb": str(capture_path),
            "route_plan": str(metadata_dir / "route_plan.json"),
        },
    )
    print(json.dumps({"status": "success" if accepted else "failed", "output_dir": str(output_dir), "metrics": metrics, "validation": validation}, ensure_ascii=False))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
