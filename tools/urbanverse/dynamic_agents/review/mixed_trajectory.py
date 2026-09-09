"""Build one overhead review image from recorded mixed-agent runtime states."""

from __future__ import annotations

import colorsys
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from urbanverse.dynamic_agents.pedestrians import WalkableRegions
from urbanverse.viz_style import load_font


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(parent: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (parent / candidate).resolve()


def _colour(index: int, total: int, *, saturation: float = 0.72) -> tuple[int, int, int]:
    hue = (index / max(1, total) + 0.07) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, 0.92)
    return int(red * 255), int(green * 255), int(blue * 255)


def _valid_xy(value: Iterable[float]) -> np.ndarray | None:
    point = np.asarray(tuple(value), dtype=np.float64)
    if point.shape[0] < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return point[:2]


def _append_track(
    tracks: dict[str, list[np.ndarray]], agent_id: str, value: Iterable[float]
) -> None:
    point = _valid_xy(value)
    if point is not None:
        tracks[agent_id].append(point)


def load_actual_tracks(trajectory_path: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    """Load only runtime positions; planned paths are intentionally excluded."""

    tracks: dict[str, dict[str, list[np.ndarray]]] = {
        "go2": defaultdict(list),
        "vehicle": defaultdict(list),
        "pedestrian": defaultdict(list),
        "micromobility": defaultdict(list),
    }
    with Path(trajectory_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            segment = int(row.get('motion_segment_id', 0))
            _append_track(tracks["go2"], f"Go2_s{segment}", row["base_position_world"])
            traffic = row.get("scene10_multivehicle_traffic") or {}
            for agent in traffic.get("agents", []):
                if not agent.get("active_on_road", False):
                    continue
                _append_track(
                    tracks["vehicle"],
                    f"V{int(agent['id']):02d}",
                    agent["center_xyz"],
                )
            people_ids = row.get("pedestrian_ids", [])
            for index, position in enumerate(row.get("pedestrian_positions_xyz", [])):
                raw_id = people_ids[index] if index < len(people_ids) else f"{index:02d}"
                label = str(raw_id)
                if not label.startswith("P"):
                    digits = "".join(character for character in label if character.isdigit())
                    label = f"P{int(digits):02d}" if digits else f"P{index:02d}"
                _append_track(tracks["pedestrian"], label, position)
            for agent_id, state in row.get("micromobility_states", {}).items():
                label = str(agent_id)
                if not label.startswith("M"):
                    digits = "".join(character for character in label if character.isdigit())
                    label = f"M{int(digits):02d}" if digits else label
                _append_track(tracks["micromobility"], label, state["position_xy"])
    return {kind: dict(values) for kind, values in tracks.items()}


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    colour: tuple[int, int, int],
) -> None:
    if len(points) < 2:
        return
    index = max(1, len(points) - max(2, len(points) // 8))
    start = np.asarray(points[index - 1], dtype=np.float64)
    end = np.asarray(points[index], dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        start = np.asarray(points[0], dtype=np.float64)
        end = np.asarray(points[-1], dtype=np.float64)
        delta = end - start
        length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    direction = delta / length
    perpendicular = np.asarray((-direction[1], direction[0]))
    tip = end
    left = tip - direction * 14.0 + perpendicular * 6.0
    right = tip - direction * 14.0 - perpendicular * 6.0
    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=colour)


def build_mixed_actual_trajectory_map(
    output_path: Path,
    trajectory_path: Path,
    mixed_config_path: Path,
    traffic_scene_config_path: Path,
    *, vehicle_only: bool = False,
) -> dict[str, Any]:
    """Render actual sampled trajectories and durable spatial constraints."""

    mixed_config_path = Path(mixed_config_path).resolve()
    mixed_payload = _read_json(mixed_config_path)
    regions_path = _resolve(mixed_config_path.parent, mixed_payload["walkable_regions"])
    regions = WalkableRegions.load(regions_path)

    scene_path = Path(traffic_scene_config_path).resolve()
    scene = _read_json(scene_path)
    mesh_spec = scene.get('traffic', {}).get('mesh_navigation')
    mesh_grid = None
    if mesh_spec:
        from urbanverse.dynamic_agents.navigation.mesh_loop_workflow import MeshMap
        road_path = obstacle_path = _resolve(scene_path.parent, mesh_spec['inventory'])
        mesh_grid = MeshMap(obstacle_path)
        mesh_settings = _read_json(_resolve(scene_path.parent, mesh_spec['settings']))
        road_polygons, obstacle_bodies = [], []
    else:
        road_path = _resolve(scene_path.parent, scene["road_surface"]["footprint_inventory"])
        obstacle_path = _resolve(scene_path.parent, scene["static_obstacle_inventory"])
        road_polygons = _read_json(road_path)["polygons_xy"]
        obstacle_bodies = _read_json(obstacle_path)["bodies"]
    tracks = load_actual_tracks(trajectory_path)
    if vehicle_only:
        tracks = {kind: values if kind == "vehicle" else {} for kind, values in tracks.items()}

    canvas_width, canvas_height = 2400, 1500
    map_left, map_top, map_right, map_bottom = 55, 105, 1910, 1440
    x0, y0, x1, y1 = regions.config.world_bounds_xyxy
    full_bounds = (x0, y0, x1, y1)
    if vehicle_only and tracks["vehicle"]:
        xy = np.asarray([point for values in tracks["vehicle"].values() for point in values])
        x0, y0 = xy.min(axis=0) - 12.0
        x1, y1 = xy.max(axis=0) + 12.0
        half_width = max((x1-x0)*0.5, (y1-y0)*0.30)
        center_x = (x0+x1)*0.5
        x0, x1 = center_x-half_width, center_x+half_width
    world_aspect = (x1 - x0) / (y1 - y0)
    available_width = map_right - map_left
    available_height = map_bottom - map_top
    if available_width / available_height > world_aspect:
        rendered_width = int(available_height * world_aspect)
        map_left += (available_width - rendered_width) // 2
        map_right = map_left + rendered_width
    else:
        rendered_height = int(available_width / world_aspect)
        map_top += (available_height - rendered_height) // 2
        map_bottom = map_top + rendered_height

    def pixel(value: Iterable[float]) -> tuple[int, int]:
        x, y = map(float, tuple(value)[:2])
        px = map_left + (x - x0) / (x1 - x0) * (map_right - map_left)
        py = map_top + (y1 - y) / (y1 - y0) * (map_bottom - map_top)
        return int(round(px)), int(round(py))

    canvas = Image.new("RGB", (canvas_width, canvas_height), (19, 27, 32))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(35, bold=True)
    body_font = load_font(21)
    small_font = load_font(17)
    tiny_font = load_font(14, bold=True)
    draw.text((55, 25), "车辆实际轨迹俯视图" if vehicle_only else "联合仿真实际轨迹俯视图", fill=(242, 247, 249), font=title_font)
    draw.text(
        (650, 37),
        "仅使用本次运行采样位置；箭头表示运动方向，S/E 表示起点/终点",
        fill=(187, 203, 211),
        font=body_font,
    )
    draw.rectangle((map_left, map_top, map_right, map_bottom), fill=(48, 57, 62))

    walkable = Image.fromarray((regions.admitted_mask.astype(np.uint8) * 255), mode="L")
    def crop_to_view(img: Image.Image) -> Image.Image:
        bx0, by0, bx1, by1 = full_bounds
        return img.crop((
            round((x0-bx0)/(bx1-bx0)*img.width), round((by1-y1)/(by1-by0)*img.height),
            round((x1-bx0)/(bx1-bx0)*img.width), round((by1-y0)/(by1-by0)*img.height),
        )) if vehicle_only else img
    walkable = crop_to_view(walkable)
    walkable = walkable.resize(
        (map_right - map_left, map_bottom - map_top), Image.Resampling.NEAREST
    )
    walking_fill = Image.new("RGB", walkable.size, (74, 128, 92))
    canvas.paste(walking_fill, (map_left, map_top), walkable)
    draw = ImageDraw.Draw(canvas)
    for polygon in road_polygons:
        draw.polygon([pixel(point) for point in polygon], fill=(93, 101, 106))
    # Repaint the shared walking surface transparently over the lane footprint.
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_region = Image.new("RGBA", walkable.size, (73, 181, 113, 105))
    overlay.paste(overlay_region, (map_left, map_top), walkable)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for body in obstacle_bodies:
        draw.polygon(
            [pixel(point) for point in body["corners_xy"]],
            fill=(37, 39, 42),
            outline=(243, 88, 70),
            width=2,
        )

    if mesh_grid is not None:
        if not np.allclose(mesh_grid.view, full_bounds):
            raise ValueError('Mesh/trajectory review coordinate transforms differ')
        rgb = np.full((mesh_grid.h, mesh_grid.w, 3), (48, 57, 62), dtype=np.uint8)
        rgb[mesh_grid.lane] = (93, 101, 106)
        rgb[mesh_grid.shared] = (74, 128, 92)
        rgb[mesh_grid.occupancy(mesh_settings.get('removed_instance_roots', []))] = (37, 39, 42)
        canvas.paste(crop_to_view(Image.fromarray(rgb)).resize(
            (map_right-map_left, map_bottom-map_top), Image.Resampling.NEAREST), (map_left, map_top))
        draw = ImageDraw.Draw(canvas)

    kind_settings = {
        "vehicle": (5, "V", 0.72),
        "pedestrian": (4, "P", 0.68),
        "micromobility": (4, "M", 0.84),
        "go2": (8, "Go2", 0.85),
    }
    kind_counts: dict[str, int] = {}
    point_counts: dict[str, int] = {}
    for kind in ("vehicle", "pedestrian", "micromobility", "go2"):
        entries = sorted(tracks[kind].items())
        kind_counts[kind] = int(bool(entries)) if kind == 'go2' else len(entries)
        for index, (agent_id, values) in enumerate(entries):
            points = [pixel(value) for value in values]
            if not points:
                continue
            width, _prefix, saturation = kind_settings[kind]
            colour = (255, 222, 35) if kind == "go2" else _colour(index, len(entries), saturation=saturation)
            outline = (12, 18, 21)
            if len(points) >= 2:
                draw.line(points, fill=outline, width=width + 4, joint="curve")
                draw.line(points, fill=colour, width=width, joint="curve")
            start, end = points[0], points[-1]
            radius = 7 if kind != "go2" else 10
            draw.ellipse(
                (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
                fill=(53, 224, 118),
                outline=outline,
                width=2,
            )
            draw.rectangle(
                (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
                fill=(255, 111, 86),
                outline=outline,
                width=2,
            )
            _draw_arrow(draw, points, colour)
            draw.text(
                (start[0] + 8, start[1] - 18),
                f"{agent_id} S",
                fill=colour,
                stroke_width=3,
                stroke_fill=outline,
                font=tiny_font,
            )
            draw.text(
                (end[0] + 8, end[1] + 2),
                f"{agent_id} E",
                fill=colour,
                stroke_width=3,
                stroke_fill=outline,
                font=tiny_font,
            )
            point_counts[agent_id] = len(points)

    panel_x = 1950
    draw.rounded_rectangle(
        (panel_x, 105, 2365, 1440),
        radius=16,
        fill=(28, 39, 45),
        outline=(79, 94, 102),
        width=2,
    )
    y = 135
    draw.text((panel_x + 24, y), "图层", fill=(241, 246, 248), font=body_font)
    y += 45
    legend = [
        ((73, 181, 113), "共同可行区"),
        ((93, 101, 106), "审核 Lane"),
        ((243, 88, 70), "区域内障碍物"),
        ((255, 222, 35), "Go2 实际轨迹"),
    ]
    if vehicle_only:
        legend = legend[:3]
    if mesh_grid is not None:
        legend[2] = ((37, 39, 42), "真实网格障碍区")
    for colour, label in legend:
        draw.line((panel_x + 25, y + 10, panel_x + 72, y + 10), fill=colour, width=8)
        draw.text((panel_x + 88, y - 3), label, fill=(218, 229, 234), font=small_font)
        y += 38
    y += 20
    draw.text((panel_x + 24, y), "动态物体", fill=(241, 246, 248), font=body_font)
    y += 43
    labels = {
        "vehicle": "车辆 V",
        "pedestrian": "行人 P",
        "micromobility": "二轮车 M",
        "go2": "机器狗",
    }
    for kind in ("vehicle", "pedestrian", "micromobility", "go2"):
        if vehicle_only and kind != "vehicle":
            continue
        draw.text(
            (panel_x + 27, y),
            f"{labels[kind]}：{kind_counts[kind]}",
            fill=(214, 225, 230),
            font=small_font,
        )
        y += 35
    y += 20
    draw.text((panel_x + 24, y), "端点", fill=(241, 246, 248), font=body_font)
    y += 43
    draw.ellipse((panel_x + 27, y, panel_x + 43, y + 16), fill=(53, 224, 118))
    draw.text((panel_x + 58, y - 4), "S：首次采样位置", fill=(214, 225, 230), font=small_font)
    y += 38
    draw.rectangle((panel_x + 27, y, panel_x + 43, y + 16), fill=(255, 111, 86))
    draw.text((panel_x + 58, y - 4), "E：末次采样位置", fill=(214, 225, 230), font=small_font)
    y += 58
    draw.text((panel_x + 24, y), "说明", fill=(241, 246, 248), font=body_font)
    y += 43
    for line in (
        "轨迹来自 trajectory.jsonl",
        "未用目标点或规划路径替代",
        "重叠线仍按独立 ID 标注",
        "不同车辆共用回环，轨迹会重叠" if vehicle_only else "Go2 恢复跳跃不连接为步行线",
        "非 RTX 画面；仅审阅运动路线",
    ):
        draw.text((panel_x + 27, y), line, fill=(184, 202, 210), font=small_font)
        y += 31
    north_x, north_y = panel_x + 340, 1300
    draw.polygon(
        [(north_x, north_y - 55), (north_x - 16, north_y), (north_x + 16, north_y)],
        fill=(235, 241, 244),
    )
    draw.text((north_x - 8, north_y + 8), "N", fill=(235, 241, 244), font=body_font)
    draw.rectangle((map_left, map_top, map_right, map_bottom), outline=(153, 170, 179), width=2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": str(output_path.resolve()),
        "source_trajectory": str(Path(trajectory_path).resolve()),
        "runtime_positions_only": True,
        "agent_counts": kind_counts,
        "sample_counts": point_counts,
        "world_bounds_xyxy": [float(x0), float(y0), float(x1), float(y1)],
    }
