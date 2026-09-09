"""Asset calibration contract for bicycle-like dynamic agents."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "configs/dynamic_agents/catalogs/micromobility_assets.json"


@dataclass(frozen=True)
class AssetCalibration:
    asset_id: str
    agent_class: str
    source_glb: str
    source_sha256: str
    license: str
    scale_m_per_asset_unit: float
    isaac45_visual_scale: float
    source_up_axis: str
    source_front_axis: tuple[float, float, float]
    converted_front_axis_xy: tuple[float, float]
    visible_aabb_min_xyz_m: tuple[float, float, float]
    visible_aabb_max_xyz_m: tuple[float, float, float]
    visible_center_xyz_m: tuple[float, float, float]
    root_to_visible_center_xyz_m: tuple[float, float, float]
    ground_offset_m: float
    length_m: float
    width_m: float
    height_m: float
    wheelbase_m: float
    minimum_turning_radius_m: float
    maximum_steering_rad: float

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "AssetCalibration":
        bounds = row["visible_geometry"]
        kinematics = row["kinematics"]
        grounding = row["grounding"]
        axes = row["axes"]
        value = cls(
            asset_id=str(row["asset_id"]),
            agent_class=str(row["agent_class"]),
            source_glb=str(row["source_glb"]),
            source_sha256=str(row["source_sha256"]),
            license=str(row["license"]),
            scale_m_per_asset_unit=float(row["scale_m_per_asset_unit"]),
            isaac45_visual_scale=float(row["isaac45_conversion"]["run_local_visual_scale"]),
            source_up_axis=str(axes["source_up_axis"]),
            source_front_axis=tuple(float(v) for v in axes["source_front_axis"]),
            converted_front_axis_xy=tuple(float(v) for v in axes["converted_front_axis_xy"]),
            visible_aabb_min_xyz_m=tuple(float(v) for v in bounds["source_aabb_min_xyz_m"]),
            visible_aabb_max_xyz_m=tuple(float(v) for v in bounds["source_aabb_max_xyz_m"]),
            visible_center_xyz_m=tuple(float(v) for v in bounds["source_visible_center_xyz_m"]),
            root_to_visible_center_xyz_m=tuple(float(v) for v in bounds["source_root_to_visible_center_xyz_m"]),
            ground_offset_m=float(grounding["converted_root_z_above_support_m"]),
            length_m=float(bounds["calibrated_length_m"]),
            width_m=float(bounds["calibrated_width_m"]),
            height_m=float(bounds["calibrated_height_m"]),
            wheelbase_m=float(kinematics["wheelbase_m"]),
            minimum_turning_radius_m=float(kinematics["minimum_turning_radius_m"]),
            maximum_steering_rad=float(kinematics["maximum_steering_rad"]),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.agent_class not in {"bicycle", "electric_two_wheeler"}:
            raise ValueError(f"unsupported micromobility class: {self.agent_class}")
        if self.scale_m_per_asset_unit <= 0.0:
            raise ValueError("asset scale must be positive")
        if self.isaac45_visual_scale <= 0.0:
            raise ValueError("Isaac 4.5 run-local visual scale must be positive")
        if min(self.length_m, self.width_m, self.height_m, self.wheelbase_m) <= 0.0:
            raise ValueError("calibrated dimensions and wheelbase must be positive")
        if self.length_m <= self.width_m:
            raise ValueError("two-wheel length must exceed its width")
        if not 0.45 * self.length_m <= self.wheelbase_m <= self.length_m:
            raise ValueError("wheelbase is inconsistent with visible length")
        expected_radius = self.wheelbase_m / math.tan(self.maximum_steering_rad)
        if self.minimum_turning_radius_m + 1.0e-6 < expected_radius:
            raise ValueError("minimum turning radius violates wheelbase/steering bound")
        norm = math.hypot(*self.converted_front_axis_xy)
        if abs(norm - 1.0) > 1.0e-6:
            raise ValueError("converted front axis must be a unit XY vector")
        if abs(self.visible_aabb_min_xyz_m[1]) > 1.0e-4:
            raise ValueError("source asset audit expects Y-up geometry resting at Y=0")
        if abs(self.visible_center_xyz_m[1] - self.height_m * 0.5) > 0.03:
            raise ValueError("visible center/height/ground calibration is inconsistent")


def load_catalog(path: Path) -> dict[str, AssetCalibration]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [AssetCalibration.from_dict(row) for row in payload["records"]]
    if len({row.asset_id for row in records}) != len(records):
        raise ValueError("duplicate asset_id in micromobility catalog")
    return {row.asset_id: row for row in records}
