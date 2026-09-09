"""Strict, simulator-independent configuration contract for the A/B experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def _vec(payload: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(payload, list) or len(payload) != length:
        raise ValueError(f"{label} must be a {length}-element list")
    result = tuple(float(value) for value in payload)
    if not all(value == value and abs(value) != float("inf") for value in result):
        raise ValueError(f"{label} must contain finite values")
    return result


@dataclass(frozen=True)
class ObstacleConfig:
    obstacle_id: str
    category: str
    authored_prim_path: str
    runtime_prim_path: str
    center_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]
    yaw_deg: float
    display_color_rgb: tuple[float, float, float]
    robot_proxy_radius_m: float
    shape: str
    visual_asset_relative: str | None
    visual_asset_kind: str | None
    visual_up_axis: str | None
    visual_fit_fraction: float
    visual_fit_mode: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ObstacleConfig":
        obstacle = cls(
            obstacle_id=str(payload["obstacle_id"]),
            category=str(payload["category"]),
            authored_prim_path=str(payload["authored_prim_path"]),
            runtime_prim_path=str(payload["runtime_prim_path"]),
            center_xyz=_vec(payload["center_xyz"], 3, "obstacle.center_xyz"),
            dimensions_xyz=_vec(payload["dimensions_xyz"], 3, "obstacle.dimensions_xyz"),
            yaw_deg=float(payload.get("yaw_deg", 0.0)),
            display_color_rgb=_vec(
                payload.get("display_color_rgb", [0.32, 0.34, 0.37]),
                3,
                "obstacle.display_color_rgb",
            ),
            robot_proxy_radius_m=float(payload.get("robot_proxy_radius_m", 0.42)),
            shape=str(payload.get("shape", "cuboid")),
            visual_asset_relative=(
                str(payload["visual_asset_relative"])
                if payload.get("visual_asset_relative") is not None
                else None
            ),
            visual_asset_kind=(
                str(payload["visual_asset_kind"])
                if payload.get("visual_asset_kind") is not None
                else None
            ),
            visual_up_axis=(
                str(payload["visual_up_axis"])
                if payload.get("visual_up_axis") is not None
                else None
            ),
            visual_fit_fraction=float(payload.get("visual_fit_fraction", 0.92)),
            visual_fit_mode=str(payload.get("visual_fit_mode", "all_axes")),
        )
        if not obstacle.obstacle_id or not obstacle.category:
            raise ValueError("obstacle_id and category must be non-empty")
        if not obstacle.authored_prim_path.startswith(
            "/World/envs/env_.*/CollisionPassthrough"
        ):
            raise ValueError("authored obstacle must use the owned per-environment namespace")
        if not obstacle.runtime_prim_path.startswith(
            "/World/envs/env_0/CollisionPassthrough"
        ):
            raise ValueError("runtime obstacle must use the owned env_0 namespace")
        if min(obstacle.dimensions_xyz) <= 0.0 or obstacle.robot_proxy_radius_m <= 0.0:
            raise ValueError("obstacle dimensions and robot proxy radius must be positive")
        if any(not 0.0 <= value <= 1.0 for value in obstacle.display_color_rgb):
            raise ValueError("display_color_rgb must be in [0, 1]")
        if obstacle.shape not in {"cuboid", "capsule", "cylinder"}:
            raise ValueError(f"unsupported obstacle shape: {obstacle.shape}")
        visual_fields = (
            obstacle.visual_asset_relative,
            obstacle.visual_asset_kind,
            obstacle.visual_up_axis,
        )
        if any(value is not None for value in visual_fields) and not all(
            value is not None for value in visual_fields
        ):
            raise ValueError("semantic visual requires asset, kind, and up axis")
        if obstacle.visual_asset_kind not in {None, "usd_reference", "gltf_payload"}:
            raise ValueError(f"unsupported visual asset kind: {obstacle.visual_asset_kind}")
        if obstacle.visual_up_axis not in {None, "z_up", "y_up"}:
            raise ValueError(f"unsupported visual up axis: {obstacle.visual_up_axis}")
        if not 0.0 < obstacle.visual_fit_fraction <= 1.0:
            raise ValueError("visual_fit_fraction must be in (0, 1]")
        if obstacle.visual_fit_mode not in {"all_axes", "height"}:
            raise ValueError(f"unsupported visual fit mode: {obstacle.visual_fit_mode}")
        return obstacle

    def visual_asset(self, asset_root: Path) -> Path | None:
        if self.visual_asset_relative is None:
            return None
        return (asset_root / self.visual_asset_relative).resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    scene_id: str
    source_usd_relative: str
    source_tar_relative: str
    route_points_xy: tuple[tuple[float, float], ...]
    ground_z_m: float
    obstacle: ObstacleConfig
    obstacles: tuple[ObstacleConfig, ...]
    filter_strategy: str
    locomotion_ground_whitelist: tuple[str, ...]
    robot_prim_path: str
    protected_collision_prefixes: tuple[str, ...]
    duration_s: float
    warmup_s: float
    capture_fps: float
    resolution: tuple[int, int]
    camera_eye_xyz: tuple[float, float, float]
    camera_target_xyz: tuple[float, float, float]
    camera_focal_length_mm: float
    controller: dict[str, float]
    thresholds: dict[str, float]
    source_path: Path
    sha256: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any], source_path: Path) -> "ExperimentConfig":
        route = tuple(_vec(row, 2, "route_points_xy[]") for row in payload["route_points_xy"])
        if len(route) < 2:
            raise ValueError("route_points_xy needs at least two points")
        raw = source_path.read_bytes()
        obstacle_payloads = payload.get("obstacles")
        if obstacle_payloads is None:
            obstacle_payloads = [payload["obstacle"]]
        obstacles = tuple(ObstacleConfig.from_payload(row) for row in obstacle_payloads)
        config = cls(
            schema_version=int(payload["schema_version"]),
            experiment_id=str(payload["experiment_id"]),
            scene_id=str(payload["scene_id"]),
            source_usd_relative=str(payload["source_usd_relative"]),
            source_tar_relative=str(payload["source_tar_relative"]),
            route_points_xy=route,
            ground_z_m=float(payload["ground_z_m"]),
            obstacle=obstacles[0],
            obstacles=obstacles,
            filter_strategy=str(payload.get("filter_strategy", "selected_pair")),
            locomotion_ground_whitelist=tuple(
                str(value)
                for value in payload.get(
                    "locomotion_ground_whitelist", ["/World/Go2RuntimeSupportPlane"]
                )
            ),
            robot_prim_path=str(payload.get("robot_prim_path", "/World/envs/env_0/Robot")),
            protected_collision_prefixes=tuple(
                str(value) for value in payload["protected_collision_prefixes"]
            ),
            duration_s=float(payload.get("duration_s", 30.0)),
            warmup_s=float(payload.get("warmup_s", 1.0)),
            capture_fps=float(payload.get("capture_fps", 10.0)),
            resolution=tuple(int(value) for value in _vec(payload["resolution"], 2, "resolution")),
            camera_eye_xyz=_vec(payload["camera_eye_xyz"], 3, "camera_eye_xyz"),
            camera_target_xyz=_vec(payload["camera_target_xyz"], 3, "camera_target_xyz"),
            camera_focal_length_mm=float(payload.get("camera_focal_length_mm", 24.0)),
            controller={key: float(value) for key, value in payload["controller"].items()},
            thresholds={key: float(value) for key, value in payload["thresholds"].items()},
            source_path=source_path.resolve(),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if not self.experiment_id or not self.scene_id:
            raise ValueError("experiment_id and scene_id must be non-empty")
        if self.robot_prim_path != "/World/envs/env_0/Robot/base":
            raise ValueError("this experiment requires Isaac Lab's read-back Go2 articulation root")
        if self.filter_strategy not in {"selected_pair", "world_collision_groups"}:
            raise ValueError(f"unsupported filter_strategy: {self.filter_strategy}")
        if not self.obstacles:
            raise ValueError("at least one obstacle is required")
        if len({row.obstacle_id for row in self.obstacles}) != len(self.obstacles):
            raise ValueError("obstacle ids must be unique")
        if len({row.runtime_prim_path for row in self.obstacles}) != len(self.obstacles):
            raise ValueError("obstacle runtime prim paths must be unique")
        if self.filter_strategy == "selected_pair" and len(self.obstacles) != 1:
            raise ValueError("selected_pair strategy requires exactly one obstacle")
        if self.filter_strategy == "world_collision_groups":
            if len(self.obstacles) < 3:
                raise ValueError("world collision-group qualification requires at least three obstacles")
            if self.locomotion_ground_whitelist != ("/World/Go2RuntimeSupportPlane",):
                raise ValueError("world collision-group qualification requires the explicit support-plane whitelist")
        required_protected = (
            ("/World/ground", "/World/Go2RuntimeSupportPlane", "/World/envs/env_0/Robot")
            if self.filter_strategy == "selected_pair"
            else ("/World/Go2RuntimeSupportPlane", "/World/envs/env_0/Robot")
        )
        for prefix in required_protected:
            if prefix not in self.protected_collision_prefixes:
                raise ValueError(f"missing protected collision prefix: {prefix}")
        # The obstacle is registered through InteractiveSceneCfg before reset,
        # so Fabric, RTX and PhysX consume the same explicit asset.
        if not self.obstacle.runtime_prim_path.startswith(
            "/World/envs/env_0/CollisionPassthrough"
        ):
            raise ValueError("selected obstacle is outside its owned runtime namespace")
        if self.duration_s <= 0.0 or self.warmup_s < 0.0 or self.capture_fps <= 0.0:
            raise ValueError("duration/capture values are invalid")
        if min(self.resolution) <= 0:
            raise ValueError("resolution must be positive")
        if self.camera_focal_length_mm <= 0.0:
            raise ValueError("camera_focal_length_mm must be positive")
        required_controller = {
            "lookahead_distance_m",
            "max_forward_speed_mps",
            "minimum_tracking_speed_mps",
            "max_tracking_yaw_rate_radps",
        }
        required_thresholds = {
            "goal_tolerance_m",
            "minimum_passthrough_foot_support_fraction",
            "maximum_step_displacement_m",
            "maximum_cross_track_error_m",
        }
        if not required_controller.issubset(self.controller):
            raise ValueError("controller thresholds are incomplete")
        if not required_thresholds.issubset(self.thresholds):
            raise ValueError("acceptance thresholds are incomplete")

    def source_usd(self, asset_root: Path) -> Path:
        return (asset_root / self.source_usd_relative).resolve()

    def source_tar(self, asset_root: Path) -> Path:
        return (asset_root / self.source_tar_relative).resolve()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "scene_id": self.scene_id,
            "config_path": str(self.source_path),
            "config_sha256": self.sha256,
            "filter_strategy": self.filter_strategy,
            "locomotion_ground_whitelist": list(self.locomotion_ground_whitelist),
            "obstacles": [
                {
                    "obstacle_id": obstacle.obstacle_id,
                    "category": obstacle.category,
                    "shape": obstacle.shape,
                    "authored_prim_path": obstacle.authored_prim_path,
                    "runtime_prim_path": obstacle.runtime_prim_path,
                    "center_xyz": list(obstacle.center_xyz),
                    "dimensions_xyz": list(obstacle.dimensions_xyz),
                    "yaw_deg": obstacle.yaw_deg,
                    "visual_asset_relative": obstacle.visual_asset_relative,
                    "visual_asset_kind": obstacle.visual_asset_kind,
                    "visual_up_axis": obstacle.visual_up_axis,
                    "visual_fit_fraction": obstacle.visual_fit_fraction,
                    "visual_fit_mode": obstacle.visual_fit_mode,
                }
                for obstacle in self.obstacles
            ],
            "robot_prim_path": self.robot_prim_path,
            "protected_collision_prefixes": list(self.protected_collision_prefixes),
            "route_points_xy": [list(row) for row in self.route_points_xy],
            "ground_z_m": self.ground_z_m,
            "camera_focal_length_mm": self.camera_focal_length_mm,
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return ExperimentConfig.from_payload(payload, source_path)
