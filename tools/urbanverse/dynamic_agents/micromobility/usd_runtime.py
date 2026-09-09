"""Isaac 4.5 conversion and visible USD transforms for micromobility agents."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibration import AssetCalibration
from .motion import MicromobilityState, visible_center_and_root_pose
from .roaming import MicromobilityAgentSpec, attach_official_people_dynamic_obstacle


def choose_support_height(
    samples_z_m: Sequence[float],
    *,
    fallback_z_m: float,
    maximum_delta_m: float = 0.50,
) -> float:
    """Choose a local surface height while rejecting overhead ray hits."""

    accepted = [
        float(value)
        for value in samples_z_m
        if math.isfinite(float(value))
        and abs(float(value) - float(fallback_z_m)) <= float(maximum_delta_m)
    ]
    return float(np.median(accepted)) if accepted else float(fallback_z_m)


def convert_assets(
    app: Any,
    catalog: dict[str, AssetCalibration],
    data_root: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Convert only catalog assets used by the run into run-local USDs."""

    import omni.kit.asset_converter
    from omni.kit.async_engine import run_coroutine

    converted: dict[str, Path] = {}
    for asset_id, calibration in catalog.items():
        source = (Path(data_root) / calibration.source_glb).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = Path(output_root) / asset_id / "asset.usd"
        target.parent.mkdir(parents=True, exist_ok=True)
        context = omni.kit.asset_converter.AssetConverterContext()
        context.keep_all_materials = True
        context.create_world_as_default_root_prim = True
        context.use_meter_as_world_unit = True
        task = omni.kit.asset_converter.get_instance().create_converter_task(
            str(source), str(target), None, context
        )
        future = run_coroutine(task.wait_until_finished())
        while not future.done():
            app.update()
        if not future.result() or not target.is_file():
            raise RuntimeError(f"conversion failed for {asset_id}: {task.get_error_message()}")
        converted[asset_id] = target
    return converted


class MicromobilityUsdRuntime:
    """Render-only whole-body USD placement; kinematics stay in the manager."""

    def __init__(
        self,
        stage: Any,
        specs: Sequence[MicromobilityAgentSpec],
        catalog: dict[str, AssetCalibration],
        converted_assets: dict[str, Path],
        *,
        ground_z_m: float,
        parent_path: str = "/World/Micromobility",
        register_official_dynamic_obstacles: bool = True,
        scene_query: Any | None = None,
        support_query_interval_m: float = 0.20,
    ) -> None:
        from pxr import Gf, UsdGeom

        self.stage = stage
        self.catalog = catalog
        self.specs = {spec.agent_id: spec for spec in specs}
        self.ground_z_m = float(ground_z_m)
        self.scene_query = scene_query
        self.support_query_interval_m = float(support_query_interval_m)
        self.support_query_positions: dict[str, np.ndarray] = {}
        self.current_support_z_m: dict[str, float] = {}
        self.support_query_count = 0
        self.support_query_hit_count = 0
        self.support_query_fallback_count = 0
        self.support_query_paths: set[str] = set()
        self.parent_path = str(parent_path).rstrip("/")
        self.ops = {}
        self.dynamic_obstacle_scripts = {}
        self.official_obstacle_radii_m = {}
        UsdGeom.Scope.Define(stage, parent_path)
        for spec in specs:
            root = UsdGeom.Xform.Define(stage, f"{parent_path}/{spec.agent_id}")
            root.ClearXformOpOrder()
            self.ops[spec.agent_id] = root.AddTransformOp()
            visual = stage.DefinePrim(f"{parent_path}/{spec.agent_id}/Visual", "Xform")
            visual.GetReferences().AddReference(str(converted_assets[spec.asset_id]))
            visual.Load()
            visual_xform = UsdGeom.Xformable(visual)
            visual_xform.ClearXformOpOrder()
            visual_xform.AddRotateXOp().Set(90.0)
            scale = float(catalog[spec.asset_id].isaac45_visual_scale)
            visual_xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
            calibration = catalog[spec.asset_id]
            self.official_obstacle_radii_m[spec.agent_id] = (
                math.hypot(calibration.length_m * 0.5, calibration.width_m * 0.5) + 0.30
            )
            if register_official_dynamic_obstacles:
                self.dynamic_obstacle_scripts[spec.agent_id] = (
                    attach_official_people_dynamic_obstacle(root.GetPrim())
                )
        self.register_official_dynamic_obstacles = bool(
            register_official_dynamic_obstacles
        )

    def _support_height(self, agent_id: str, position_xy: np.ndarray) -> float:
        previous_xy = self.support_query_positions.get(agent_id)
        if (
            agent_id in self.current_support_z_m
            and previous_xy is not None
            and float(np.linalg.norm(position_xy - previous_xy))
            < self.support_query_interval_m
        ):
            return self.current_support_z_m[agent_id]
        self.support_query_positions[agent_id] = position_xy.copy()
        if self.scene_query is None:
            self.current_support_z_m[agent_id] = self.ground_z_m
            return self.ground_z_m

        import carb

        offsets = (
            (0.0, 0.0),
            (0.22, 0.0),
            (-0.22, 0.0),
            (0.0, 0.22),
            (0.0, -0.22),
        )
        samples: list[float] = []
        self.support_query_count += len(offsets)
        for offset_x, offset_y in offsets:
            ray = self.scene_query.raycast_closest(
                carb.Float3(
                    float(position_xy[0] + offset_x),
                    float(position_xy[1] + offset_y),
                    self.ground_z_m + 4.0,
                ),
                carb.Float3(0.0, 0.0, -1.0),
                8.0,
            )
            if (
                not isinstance(ray, dict)
                or not bool(ray.get("hit", False))
                or ray.get("position") is None
            ):
                continue
            value = float(ray["position"][2])
            if abs(value - self.ground_z_m) <= 0.50:
                samples.append(value)
                self.support_query_paths.add(str(ray.get("collision") or ""))
        self.support_query_hit_count += len(samples)
        if not samples:
            self.support_query_fallback_count += 1
        support_z = choose_support_height(
            samples,
            fallback_z_m=self.ground_z_m,
            maximum_delta_m=0.50,
        )
        self.current_support_z_m[agent_id] = support_z
        return support_z

    def update(self, states: dict[str, MicromobilityState]) -> None:
        from pxr import Gf

        official_manager = None
        if self.register_official_dynamic_obstacles:
            import carb
            from omni.anim.people.scripts.global_character_position_manager import (
                GlobalCharacterPositionManager,
            )

            official_manager = GlobalCharacterPositionManager.get_instance()

        for agent_id, state in states.items():
            spec = self.specs[agent_id]
            support_z_m = self._support_height(agent_id, state.position_xy)
            pose = visible_center_and_root_pose(
                state, self.catalog[spec.asset_id], support_z_m
            )
            matrix = Gf.Matrix4d(1.0)
            matrix.SetRotate(
                Gf.Rotation(
                    Gf.Vec3d(0.0, 0.0, 1.0),
                    math.degrees(float(pose["root_yaw_rad"])),
                )
            )
            matrix.SetTranslateOnly(
                Gf.Vec3d(*map(float, pose["root_translation_xyz_m"]))
            )
            self.ops[agent_id].Set(matrix)
            # The stock BehaviorScript derives its radius from a composed USD
            # bbox and publishes position/future/radius in three separate
            # writes.  Converted GLB bounds and script scheduling can leave a
            # transient position entry without a radius.  Publish the same
            # official interface atomically from calibrated geometry every
            # controller update; the stock script remains attached and no ORCA
            # or private navigation steering is introduced.
            if official_manager is not None:
                path = f"{self.parent_path}/{agent_id}"
                current = carb.Float3(float(state.x_m), float(state.y_m), support_z_m)
                velocity = state.velocity_xy
                future = carb.Float3(
                    float(state.x_m + velocity[0]),
                    float(state.y_m + velocity[1]),
                    support_z_m,
                )
                official_manager.set_character_current_pos(path, current)
                official_manager.set_character_future_pos(path, future)
                official_manager.set_character_radius(
                    path, float(self.official_obstacle_radii_m[agent_id])
                )

    def visual_bounds(self) -> list[dict[str, Any]]:
        """Measure composed render bounds against the configured support Z."""

        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        rows: list[dict[str, Any]] = []
        for agent_id, spec in self.specs.items():
            root = self.stage.GetPrimAtPath(f"{self.parent_path}/{agent_id}")
            aligned = cache.ComputeWorldBound(root).ComputeAlignedRange()
            minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
            maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
            rows.append(
                {
                    "agent_id": agent_id,
                    "asset_id": spec.asset_id,
                    "minimum_xyz_m": minimum.tolist(),
                    "maximum_xyz_m": maximum.tolist(),
                    "size_xyz_m": (maximum - minimum).tolist(),
                    "configured_fallback_support_z_m": self.ground_z_m,
                    "queried_support_z_m": self.current_support_z_m.get(
                        agent_id, self.ground_z_m
                    ),
                    "visible_bottom_support_error_m": float(
                        minimum[2]
                        - self.current_support_z_m.get(agent_id, self.ground_z_m)
                    ),
                    "finite": bool(
                        np.all(np.isfinite(minimum))
                        and np.all(np.isfinite(maximum))
                    ),
                }
            )
        return rows

    def support_query_summary(self) -> dict[str, Any]:
        return {
            "method": (
                "five-point PhysX down-rays with median/outlier rejection"
                if self.scene_query is not None
                else "configured visual support height"
            ),
            "fallback_support_z_m": self.ground_z_m,
            "current_support_z_m": dict(self.current_support_z_m),
            "query_count": self.support_query_count,
            "accepted_hit_count": self.support_query_hit_count,
            "fallback_count": self.support_query_fallback_count,
            "collision_paths": sorted(self.support_query_paths),
        }
