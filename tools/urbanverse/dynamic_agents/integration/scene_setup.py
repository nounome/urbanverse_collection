"""Reusable Isaac Lab scene setup for portable traffic and Go2 routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..traffic.multivehicle_manager import DIVERSE_TRAFFIC_VEHICLE_IDS


def author_portable_traffic_visuals(
    wrapper_path: Path,
    registry_path: Path,
    vehicle_count: int,
    asset_ids: Sequence[str] = DIVERSE_TRAFFIC_VEHICLE_IDS,
    *,
    author_root: str = "/UrbanVerseAsset",
    mount_root: str = "/World/ground/terrain",
) -> list[dict[str, str]]:
    """Compose normalized vehicle USDs into a run wrapper before ``gym.make``.

    ``author_root`` is the wrapper prim that receives the portable assets and
    ``mount_root`` is its corresponding path in the final stage.  Legacy Isaac
    Lab runs author under ``/UrbanVerseAsset`` and mount it below
    ``/World/ground/terrain``.  Official Navigation/Recast runs author and mount
    directly below the native ``/World``.  Keeping both mappings explicit avoids
    baking a Scene09-only hierarchy into the reusable traffic component.
    """
    from pxr import Usd, UsdGeom

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if registry.get("catalog_kind") != "portable_direct_payload_vehicles":
        return []
    source_scene_usd = Path(registry["source_scene_usd"]).resolve()
    converted_cache_dir = Path(registry["converted_usd_cache_dir"]).resolve()
    records = {row["asset_id"]: row for row in registry["records"]}
    selected = [asset_ids[index % len(asset_ids)] for index in range(vehicle_count)]
    stage = Usd.Stage.Open(str(wrapper_path))
    if stage is None:
        raise RuntimeError(f"could not open wrapper for portable traffic: {wrapper_path}")
    stage.SetEditTarget(stage.GetRootLayer())
    configured: list[dict[str, str]] = []
    for index, asset_id in enumerate(selected):
        if asset_id not in records:
            raise KeyError(f"portable vehicle is absent from catalog: {asset_id}")
        source_path = str(records[asset_id]["scene_prim_path"])
        local_path = (
            f"{author_root.rstrip('/')}/PortableTrafficVehicle_{index:02d}"
        )
        prim = UsdGeom.Xform.Define(stage, local_path).GetPrim()
        converted_usd = converted_cache_dir / asset_id / "vehicle.usd"
        if not converted_usd.is_file():
            raise RuntimeError(
                f"portable USD cache is missing {asset_id}: {converted_usd}; "
                "run tools/dynamic_agents/runners/run_vehicle_catalog_conversion.sh"
            )
        prim.GetReferences().AddReference(str(converted_usd))
        configured.append(
            {
                "asset_id": asset_id,
                "source_scene_prim": f"{source_scene_usd}{source_path}",
                "converted_usd": str(converted_usd),
                "controlled_prim": mount_root.rstrip("/")
                + local_path[len(author_root.rstrip("/")) :],
                "creation_phase": "portable source prim composed before InteractiveScene setup",
            }
        )
    stage.GetRootLayer().Save()
    return configured


def configure_preauthored_traffic_visuals(
    cfg: Any, configured: list[dict[str, str]]
) -> None:
    """Register precomposed traffic prims before Isaac Lab initializes Fabric."""
    from isaaclab.assets import AssetBaseCfg

    for index, row in enumerate(configured):
        setattr(
            cfg.scene,
            f"dynamic_traffic_vehicle_{index:02d}",
            AssetBaseCfg(
                prim_path=row["controlled_prim"],
                spawn=None,
                collision_group=-1,
            ),
        )


def configure_go2_support_corridor(
    cfg: Any,
    route_payload: dict[str, Any],
    *,
    ghost_passthrough: bool = False,
    isolate_source_scene_collisions: bool = False,
) -> dict[str, Any] | None:
    """Add an invisible flat support plane beneath an explicitly audited route."""
    support = route_payload.get("runtime_physics_support", {})
    if not bool(support.get("enabled", False)):
        return None
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils

    points = np.asarray(route_payload["points_xy"], dtype=np.float64)
    width = float(support.get("corridor_width_m", 3.0))
    endpoint_padding = float(support.get("endpoint_padding_m", 1.0))
    ground_z = float(route_payload["ground_z"])
    route_min = np.min(points, axis=0)
    route_max = np.max(points, axis=0)
    route_extent = route_max - route_min
    plane_center = 0.5 * (route_min + route_max)
    plane_size = (
        max(float(route_extent[0]) + 2.0 * endpoint_padding, width),
        max(float(route_extent[1]) + 2.0 * endpoint_padding, width),
    )
    prim_path = "/World/Go2RuntimeSupportPlane"

    # Use Isaac Lab's standard primitive spawner instead of a custom Mesh
    # callback.  The latter is safe when authoring a standalone USD helper for
    # Recast, but it can stall InteractiveScene/PhysX adoption when invoked as
    # an AssetBaseCfg spawner.  A thin cuboid has an explicit, inspectable
    # collider and places its top face exactly at the audited ground height.
    thickness = float(support.get("thickness_m", 0.04))
    spawn = sim_utils.CuboidCfg(
        size=(float(plane_size[0]), float(plane_size[1]), thickness),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=None,
    )
    if isolate_source_scene_collisions and not ghost_passthrough:
        # The official-People composition mounts the original city below
        # /UrbanVerseScene.  Some UrbanVerse modules carry coarse authored
        # colliders whose occupied volume is much larger than the reviewed
        # walking surface.  Filter only that source hierarchy from Go2 and keep
        # /World dynamic agents outside the filter.  The frozen locomotion
        # policy remains supported by this explicitly reviewed corridor.
        standard_spawn = spawn.func

        def spawn_with_source_scene_filter(
            prim_path: str,
            spawn_cfg: Any,
            translation: tuple[float, float, float] | None = None,
            orientation: tuple[float, float, float, float] | None = None,
        ) -> Any:
            prim = standard_spawn(prim_path, spawn_cfg, translation, orientation)
            from urbanverse.dynamic_agents.collision_passthrough.authoring import (
                author_go2_world_passthrough_groups,
            )

            author_go2_world_passthrough_groups(
                prim.GetStage(),
                robot_prim_path="/World/envs/env_0/Robot",
                support_paths=(prim_path,),
                obstacle_root_paths=("/UrbanVerseScene",),
            )
            return prim

        spawn.func = spawn_with_source_scene_filter

    cfg.scene.go2_runtime_support_plane = AssetBaseCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                float(plane_center[0]),
                float(plane_center[1]),
                ground_z - 0.5 * thickness,
            )
        ),
        collision_group=-1,
    )
    return {
        "enabled": True,
        "ground_z_m": ground_z,
        "corridor_width_m": width,
        "endpoint_padding_m": endpoint_padding,
        "route_bounds_xy": [route_min.tolist(), route_max.tolist()],
        "plane_center_xy": plane_center.tolist(),
        "plane_size_m": list(plane_size),
        "thickness_m": thickness,
        "segments": [prim_path],
        "mesh_path": f"{prim_path}/geometry/mesh",
        "render_visibility": "invisible",
        "ghost_passthrough": bool(ghost_passthrough),
        "source_scene_collision_isolated": bool(isolate_source_scene_collisions),
        "collision_filter_scope": (
            "Go2 versus /World except invisible support plane"
            if ghost_passthrough
            else (
                "Go2 versus /UrbanVerseScene only; /World dynamic agents retained"
                if isolate_source_scene_collisions
                else None
            )
        ),
        "scope": (
            "flat-scene locomotion support; traffic remains constrained by road polygons "
            "and OBBs"
        ),
    }
