"""Run-local USD authoring; source UrbanVerse USD files are read-only."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


WORLD_GROUP_SCOPE = "/World/CollisionPassthroughGroups"
ROBOT_COLLISION_GROUP = f"{WORLD_GROUP_SCOPE}/Go2"
OBSTACLE_COLLISION_GROUP = f"{WORLD_GROUP_SCOPE}/DefaultObstacles"


def create_wrapper(template_path: Path, source_usd: Path, output_path: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    if "SOURCE_USD" not in text:
        raise ValueError(f"wrapper template lacks SOURCE_USD token: {template_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        text.replace("SOURCE_USD", str(source_usd.resolve())), encoding="utf-8"
    )


def configure_selected_obstacle(
    cfg: Any, config: ExperimentConfig, mode: str
) -> dict[str, Any]:
    """Register one collider and, for passthrough, precompose its pair filter."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    obstacle = config.obstacle
    yaw = math.radians(float(obstacle.yaw_deg))
    spawn = sim_utils.CuboidCfg(
        size=obstacle.dimensions_xyz,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=obstacle.display_color_rgb,
            roughness=0.92,
        ),
        activate_contact_sensors=True,
    )
    standard_spawn = spawn.func

    def spawn_with_precomposed_filter(
        prim_path: str,
        spawn_cfg: Any,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
    ) -> Any:
        """Author the relation during scene construction, before PhysX starts."""
        prim = standard_spawn(prim_path, spawn_cfg, translation, orientation)
        if mode == "passthrough":
            from pxr import Sdf, UsdPhysics

            api = UsdPhysics.FilteredPairsAPI.Apply(prim)
            api.CreateFilteredPairsRel().SetTargets(
                [Sdf.Path(config.robot_prim_path)]
            )
        return prim

    spawn.func = spawn_with_precomposed_filter
    cfg.scene.collision_passthrough_obstacle = AssetBaseCfg(
        prim_path=obstacle.authored_prim_path,
        spawn=spawn,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=obstacle.center_xyz,
            rot=(
                float(math.cos(yaw / 2.0)),
                0.0,
                0.0,
                float(math.sin(yaw / 2.0)),
            ),
        ),
        collision_group=-1,
    )
    return {
        "source_usd_modified": False,
        "authored_prim_path": obstacle.authored_prim_path,
        "expected_runtime_prim_path": obstacle.runtime_prim_path,
        "category": obstacle.category,
        "obstacle_id": obstacle.obstacle_id,
        "center_xyz": list(obstacle.center_xyz),
        "dimensions_xyz": list(obstacle.dimensions_xyz),
        "yaw_deg": obstacle.yaw_deg,
        "collision_enabled": True,
        "visual_enabled": True,
        "rigid_body_kind": "kinematic",
        "filter_authoring_phase": (
            "InteractiveScene construction before PhysX startup"
            if mode == "passthrough"
            else "not authored for normal-collision baseline"
        ),
        "filter_direction": (
            "selected obstacle rigid body -> Go2 articulation root"
            if mode == "passthrough"
            else None
        ),
        "creation_phase": "registered in InteractiveSceneCfg before gym.make/reset",
    }


def author_go2_world_passthrough_groups(
    stage: Any,
    *,
    robot_prim_path: str = "/World/envs/env_0/Robot",
    support_paths: tuple[str, ...] = ("/World/Go2RuntimeSupportPlane",),
    unfiltered_world_paths: tuple[str, ...] = (),
    obstacle_root_paths: tuple[str, ...] = ("/World",),
) -> None:
    """Filter Go2 against all world colliders except approved support geometry.

    This must run from an Isaac Lab scene spawn callback before PhysX parses the
    stage.  Runtime-created vehicles and agents remain under ``/World`` and are
    therefore included automatically by the expanding collection.
    """

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    UsdGeom.Scope.Define(stage, WORLD_GROUP_SCOPE)
    robot_group = UsdPhysics.CollisionGroup.Define(stage, ROBOT_COLLISION_GROUP)
    obstacle_group = UsdPhysics.CollisionGroup.Define(stage, OBSTACLE_COLLISION_GROUP)

    robot_collection = robot_group.GetCollidersCollectionAPI()
    robot_collection.CreateExpansionRuleAttr().Set(Usd.Tokens.expandPrims)
    robot_collection.CreateIncludesRel().SetTargets([Sdf.Path(robot_prim_path)])

    obstacle_collection = obstacle_group.GetCollidersCollectionAPI()
    obstacle_collection.CreateExpansionRuleAttr().Set(Usd.Tokens.expandPrims)
    obstacle_collection.CreateIncludesRel().SetTargets(
        [Sdf.Path(path) for path in obstacle_root_paths]
    )
    obstacle_collection.CreateExcludesRel().SetTargets(
        [
            Sdf.Path(robot_prim_path),
            *[Sdf.Path(path) for path in support_paths],
            *[Sdf.Path(path) for path in unfiltered_world_paths],
        ]
    )

    robot_group.CreateFilteredGroupsRel().SetTargets([obstacle_group.GetPath()])
    obstacle_group.CreateFilteredGroupsRel().SetTargets([robot_group.GetPath()])


def _author_world_collision_groups(
    stage: Any,
    config: ExperimentConfig,
    *,
    preserve_experiment_obstacles: bool = False,
) -> None:
    """Declare the controlled Go2/world collision matrix before PhysX startup.

    Both A/B legs filter the unreviewed source scene so an unrelated source
    collider cannot launch or pin the robot at spawn.  The normal leg excludes
    the explicit experiment proxies from the *filtered world* collection, so
    Go2 still collides with those proxies and the approved support plane.  The
    passthrough leg filters the proxies as ordinary world geometry.
    """
    author_go2_world_passthrough_groups(
        stage,
        robot_prim_path="/World/envs/env_0/Robot",
        support_paths=config.locomotion_ground_whitelist,
        unfiltered_world_paths=(
            tuple(obstacle.runtime_prim_path for obstacle in config.obstacles)
            if preserve_experiment_obstacles
            else ()
        ),
        # Physical dynamic agents (vehicles and micromobility) live below
        # ``/World``.  Official IRA People are animation/navigation actors
        # below ``/Characters`` and do not author PhysX collision shapes.  Do
        # not expand that skinned hierarchy into a CollisionGroup: doing so
        # makes PhysX collection parsing scale with every skeleton prim and can
        # stall Isaac Lab environment construction.  The joint runner audits
        # this no-collider invariant after the stage has composed.
        obstacle_root_paths=("/World", "/UrbanVerseScene"),
    )


def _shape_spawn_config(sim_utils: Any, obstacle: Any) -> Any:
    common = {
        "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=obstacle.display_color_rgb,
            roughness=0.92,
        ),
        "activate_contact_sensors": True,
    }
    if obstacle.shape == "cuboid":
        return sim_utils.CuboidCfg(size=obstacle.dimensions_xyz, **common)
    if obstacle.shape == "capsule":
        return sim_utils.CapsuleCfg(
            radius=0.5 * max(obstacle.dimensions_xyz[0], obstacle.dimensions_xyz[1]),
            height=obstacle.dimensions_xyz[2],
            axis="Z",
            **common,
        )
    if obstacle.shape == "cylinder":
        return sim_utils.CylinderCfg(
            radius=0.5 * max(obstacle.dimensions_xyz[0], obstacle.dimensions_xyz[1]),
            height=obstacle.dimensions_xyz[2],
            axis="Z",
            **common,
        )
    raise ValueError(f"unsupported shape: {obstacle.shape}")


def _rotate_xyz(quaternion_xyzw: tuple[float, float, float, float], vector: Any) -> Any:
    """Rotate one XYZ vector without importing numpy into the authoring path."""
    import numpy as np

    xyz = np.asarray(quaternion_xyzw[:3], dtype=np.float64)
    scalar = float(quaternion_xyzw[3])
    value = np.asarray(vector, dtype=np.float64)
    return value + 2.0 * np.cross(xyz, np.cross(xyz, value) + scalar * value)


def _attach_semantic_visual(
    root: Any,
    obstacle: Any,
    asset_root: Path,
    visual_override: Path | None = None,
) -> dict[str, Any] | None:
    """Attach a recognizable asset while keeping one auditable hidden collider.

    ``visual_override`` lets the caller substitute a run-local converted USD for
    the source glTF: raw ``.glb`` payloads do not compose meshes in this Kit
    ("Cannot determine file format ... SDF_FORMAT_ARGS:target=usd"), so the
    ghost obstacles use the same native GLB -> USD conversion as micromobility.
    """
    if visual_override is not None:
        visual_asset = Path(visual_override)
        visual_kind = "usd_reference"
    else:
        visual_asset = obstacle.visual_asset(asset_root)
        visual_kind = obstacle.visual_asset_kind
    if visual_asset is None:
        return None
    if not visual_asset.is_file():
        raise FileNotFoundError(f"semantic obstacle visual is missing: {visual_asset}")

    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    # The primitive spawned by Isaac Lab remains the sole physical body. Hide
    # only its collision geometry, not the root, so the semantic child remains
    # visible and shares exactly the same transform and collision-group scope.
    hidden_collision_paths: list[str] = []
    for descendant in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            imageable = UsdGeom.Imageable(descendant)
            if imageable:
                imageable.MakeInvisible()
            hidden_collision_paths.append(str(descendant.GetPath()))

    stage = root.GetStage()
    visual_path = root.GetPath().AppendChild("SemanticVisual")
    visual = UsdGeom.Xform.Define(stage, visual_path).GetPrim()
    if visual_kind == "usd_reference":
        visual.GetReferences().AddReference(str(visual_asset))
    elif visual_kind == "gltf_payload":
        visual.GetPayloads().AddPayload(str(visual_asset))
    else:
        raise ValueError(f"unsupported visual kind: {visual_kind}")
    visual.Load()

    mesh_count = 0
    stripped_physics_paths: list[str] = []
    for descendant in Usd.PrimRange(visual, Usd.TraverseInstanceProxies()):
        mesh_count += int(descendant.IsA(UsdGeom.Mesh))
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            descendant.RemoveAPI(UsdPhysics.RigidBodyAPI)
            stripped_physics_paths.append(str(descendant.GetPath()))
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            descendant.RemoveAPI(UsdPhysics.CollisionAPI)
            stripped_physics_paths.append(str(descendant.GetPath()))
    if mesh_count < 1:
        raise RuntimeError(f"semantic visual composed no mesh: {visual_asset}")

    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    ).ComputeLocalBound(visual).ComputeAlignedRange()
    local_center = np.asarray(bbox.GetMidpoint(), dtype=np.float64)
    local_size = np.asarray(bbox.GetSize(), dtype=np.float64)
    if not np.all(np.isfinite(local_size)) or float(np.min(local_size)) <= 1.0e-5:
        raise RuntimeError(f"semantic visual has invalid bounds: {visual_asset}")

    # UrbanVerse glTF assets are Y-up; Isaac People USDs are already Z-up.
    quaternion_xyzw = (
        (0.5000000008902851, -0.49999999910971493, -0.49999999703181697, 0.500000002968183)
        if obstacle.visual_up_axis == "y_up"
        else (0.0, 0.0, 0.0, 1.0)
    )
    corners = np.asarray(
        [
            local_center
            + np.asarray([sx, sy, sz], dtype=np.float64) * local_size * 0.5
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    rotated_corners = np.asarray([_rotate_xyz(quaternion_xyzw, row) for row in corners])
    rotated_size = np.ptp(rotated_corners, axis=0)
    target_size = np.asarray(obstacle.dimensions_xyz, dtype=np.float64)
    uniform_scale = float(
        obstacle.visual_fit_fraction
        * (
            target_size[2] / rotated_size[2]
            if obstacle.visual_fit_mode == "height"
            else np.min(target_size / rotated_size)
        )
    )
    rotated_center = _rotate_xyz(quaternion_xyzw, local_center) * uniform_scale

    xform = UsdGeom.Xformable(visual)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "semanticCenter").Set(
        Gf.Vec3d(*(-rotated_center).tolist())
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "semanticUpAxis").Set(
        Gf.Quatd(
            float(quaternion_xyzw[3]),
            Gf.Vec3d(*map(float, quaternion_xyzw[:3])),
        )
    )
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "semanticFit").Set(
        Gf.Vec3d(uniform_scale, uniform_scale, uniform_scale)
    )
    return {
        "asset": str(visual_asset),
        "asset_kind": obstacle.visual_asset_kind,
        "source_up_axis": obstacle.visual_up_axis,
        "semantic_visual_prim_path": str(visual_path),
        "mesh_count": mesh_count,
        "source_local_center_xyz": local_center.tolist(),
        "source_local_size_xyz": local_size.tolist(),
        "oriented_size_before_fit_xyz": rotated_size.tolist(),
        "uniform_fit_scale": uniform_scale,
        "visual_fit_mode": obstacle.visual_fit_mode,
        "target_collider_dimensions_xyz": list(obstacle.dimensions_xyz),
        "hidden_collider_paths": hidden_collision_paths,
        "stripped_visual_physics_paths": sorted(set(stripped_physics_paths)),
        "physical_correspondence": "recognizable visual and hidden collider share one rigid-body root",
    }


def configure_experiment_obstacles(
    cfg: Any,
    config: ExperimentConfig,
    mode: str,
    asset_root: Path | None = None,
    visual_usd_overrides: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    """Register all run-local obstacles and the configured pre-start filter.

    ``visual_usd_overrides`` maps ``obstacle_id`` to a run-local converted USD
    that replaces a source glTF visual (raw ``.glb`` payloads do not compose
    meshes in this Kit).
    """
    if config.filter_strategy == "selected_pair":
        return [configure_selected_obstacle(cfg, config, mode)]

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    records: list[dict[str, Any]] = []
    for index, obstacle in enumerate(config.obstacles):
        yaw = math.radians(float(obstacle.yaw_deg))
        spawn = _shape_spawn_config(sim_utils, obstacle)
        standard_spawn = spawn.func
        author_groups = bool(index == len(config.obstacles) - 1)
        visual_override = (
            (visual_usd_overrides or {}).get(obstacle.obstacle_id)
            if obstacle.visual_asset_relative
            else None
        )

        def spawn_with_optional_groups(
            prim_path: str,
            spawn_cfg: Any,
            translation: tuple[float, float, float] | None = None,
            orientation: tuple[float, float, float, float] | None = None,
            _standard_spawn: Any = standard_spawn,
            _author_groups: bool = author_groups,
            _normal_collision_baseline: bool = mode == "normal",
            _obstacle: Any = obstacle,
            _visual_override: Path | None = visual_override,
        ) -> Any:
            prim = _standard_spawn(prim_path, spawn_cfg, translation, orientation)
            semantic_visual = (
                _attach_semantic_visual(prim, _obstacle, asset_root, _visual_override)
                if asset_root is not None
                else None
            )
            # Store one JSON scalar instead of a nested USD dictionary: empty
            # Python lists have no inferable Vt array type in customData.
            prim.SetCustomDataByKey(
                "collisionPassthroughSemanticVisualJson",
                json.dumps(semantic_visual or {}, sort_keys=True),
            )
            if _author_groups:
                _author_world_collision_groups(
                    prim.GetStage(),
                    config,
                    preserve_experiment_obstacles=_normal_collision_baseline,
                )
            return prim

        spawn.func = spawn_with_optional_groups
        setattr(
            cfg.scene,
            f"collision_passthrough_world_obstacle_{index:02d}",
            AssetBaseCfg(
                prim_path=obstacle.authored_prim_path,
                spawn=spawn,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=obstacle.center_xyz,
                    rot=(float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))),
                ),
                collision_group=-1,
            ),
        )
        records.append(
            {
                "obstacle_id": obstacle.obstacle_id,
                "category": obstacle.category,
                "shape": obstacle.shape,
                "authored_prim_path": obstacle.authored_prim_path,
                "expected_runtime_prim_path": obstacle.runtime_prim_path,
                "center_xyz": list(obstacle.center_xyz),
                "dimensions_xyz": list(obstacle.dimensions_xyz),
                "collision_enabled": True,
                "visual_enabled": True,
                "rigid_body_kind": "kinematic",
                "individually_named_in_filter": False,
                "go2_collision_mode": (
                    "retained for explicit A/B obstacle only"
                    if mode == "normal"
                    else "filtered as world geometry"
                ),
                "semantic_visual": (
                    str(visual_override)
                    if visual_override is not None
                    else (
                        str(obstacle.visual_asset(asset_root))
                        if asset_root is not None and obstacle.visual_asset_relative
                        else None
                    )
                ),
                "semantic_visual_kind": (
                    "usd_reference"
                    if visual_override is not None
                    else obstacle.visual_asset_kind
                ),
                "visual_collision_correspondence": (
                    "recognizable visual and hidden collider share one rigid-body root"
                    if obstacle.visual_asset_relative
                    else "primitive collider is also visible"
                ),
            }
        )
    return records


def readback_runtime_obstacle(stage: Any, config: ExperimentConfig) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    prim = stage.GetPrimAtPath(config.obstacle.runtime_prim_path)
    if not prim.IsValid():
        raise RuntimeError(
            f"run-local obstacle did not compose at {config.obstacle.runtime_prim_path}"
        )
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("run-local obstacle root lost RigidBodyAPI after composition")
    collider_rows = []
    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            collider_rows.append(
                {
                    "path": str(descendant.GetPath()),
                    "collision_enabled": bool(
                        UsdPhysics.CollisionAPI(descendant).GetCollisionEnabledAttr().Get()
                    ),
                }
            )
    if not collider_rows:
        raise RuntimeError("run-local obstacle rigid body has no collision geometry")
    if not all(row["collision_enabled"] for row in collider_rows):
        raise RuntimeError("run-local obstacle contains disabled collision geometry")
    return {
        "runtime_prim_path": config.obstacle.runtime_prim_path,
        "valid": True,
        "collision_api_on_root": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
        "collision_geometry": collider_rows,
        "collision_enabled_all": True,
        "rigid_body_api": True,
        "kinematic_enabled": bool(
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        ),
        "prim_type": prim.GetTypeName(),
    }


def readback_runtime_obstacles(stage: Any, config: ExperimentConfig) -> list[dict[str, Any]]:
    """Read every obstacle without changing the stage."""
    if len(config.obstacles) == 1:
        return [readback_runtime_obstacle(stage, config)]
    rows = []
    for obstacle in config.obstacles:
        prim = stage.GetPrimAtPath(obstacle.runtime_prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"run-local obstacle is missing: {obstacle.runtime_prim_path}")
        from pxr import Usd, UsdPhysics

        colliders = []
        for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if descendant.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(
                    {
                        "path": str(descendant.GetPath()),
                        "collision_enabled": bool(
                            UsdPhysics.CollisionAPI(descendant).GetCollisionEnabledAttr().Get()
                        ),
                    }
                )
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not colliders:
            raise RuntimeError(f"obstacle lacks rigid/collision schema: {obstacle.runtime_prim_path}")
        rows.append(
            {
                "obstacle_id": obstacle.obstacle_id,
                "category": obstacle.category,
                "shape": obstacle.shape,
                "runtime_prim_path": obstacle.runtime_prim_path,
                "rigid_body_api": True,
                "kinematic_enabled": bool(
                    UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
                ),
                "collision_geometry": colliders,
                "collision_enabled_all": all(row["collision_enabled"] for row in colliders),
                "semantic_visual": json.loads(
                    prim.GetCustomDataByKey(
                        "collisionPassthroughSemanticVisualJson"
                    )
                    or "{}"
                ),
            }
        )
    return rows
