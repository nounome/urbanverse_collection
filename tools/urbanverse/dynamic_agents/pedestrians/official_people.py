"""Animated NVIDIA People assets beside UrbanVerse traffic and Go2."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def people_visual_yaw_from_route_heading(route_heading_rad: float) -> float:
    """Convert a world +X mathematical heading to the People rig yaw.

    NVIDIA People target rigs declare ``controlRig:forwardAxis = \"MINUS Y\"``
    and ``controlRig:upAxis = \"Z\"``.  A local -Y facing vector therefore
    needs a yaw of ``pi / 2 + heading`` to point along the route direction.
    Keeping this conversion separate makes the otherwise easy-to-miss 180
    degree convention mismatch directly unit-testable.
    """
    return math.pi / 2.0 + float(route_heading_rad)


def advance_walk_animation_clock(
    current_time_s: float,
    *,
    dt_s: float,
    actual_speed_mps: float,
    preferred_speed_mps: float,
    stop_threshold_mps: float,
) -> float:
    """Advance an in-place walk clip only while its root is really moving.

    Scaling the clip clock by actual/preferred speed also keeps a yielding
    character's foot cadence coupled to its slowed root motion.  At a hard
    stop the current pose is held instead of continuing to walk in place.
    """

    if dt_s <= 0.0 or preferred_speed_mps <= 0.0 or stop_threshold_mps < 0.0:
        raise ValueError("animation clock inputs are invalid")
    speed = max(0.0, float(actual_speed_mps))
    if speed <= float(stop_threshold_mps):
        return float(current_time_s)
    return float(current_time_s) + float(dt_s) * min(
        speed / float(preferred_speed_mps), 1.5
    )


def _point_at_route_fraction(
    route: np.ndarray, fraction: float
) -> tuple[np.ndarray, int]:
    """Return a route position and the next waypoint at an arc-length fraction."""
    segment_lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    distance = float(np.clip(fraction, 0.0, 1.0)) * total
    cumulative = 0.0
    for index, segment_length in enumerate(segment_lengths):
        if distance <= cumulative + float(segment_length) or index == len(segment_lengths) - 1:
            alpha = (distance - cumulative) / max(float(segment_length), 1.0e-9)
            return route[index] + alpha * (route[index + 1] - route[index]), index + 1
        cumulative += float(segment_length)
    return route[-1].copy(), len(route) - 1


def _expand_continuous_route_config(path: Path, payload: dict[str, Any]) -> None:
    inventory_path = Path(payload["route_inventory"])
    inventory_path = (
        inventory_path if inventory_path.is_absolute() else path.parent / inventory_path
    ).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    routes = inventory.get("routes", [])
    if not routes:
        raise ValueError("continuous pedestrian route inventory contains no routes")
    assets_root = Path(payload["assets_root"])
    assets_root = (
        assets_root if assets_root.is_absolute() else path.parent / assets_root
    ).resolve()
    ground_z = float(inventory["ground_z_m"])
    slots_per_route = int(payload.get("slots_per_route", 4))
    initial_fractions = [
        float(value) for value in payload.get("initial_arc_fractions", [0.0])
    ]
    if slots_per_route < len(initial_fractions) or slots_per_route < 1:
        raise ValueError("slots_per_route must cover every initial pedestrian")
    expanded: list[dict[str, Any]] = []
    route_centerlines: list[tuple[str, np.ndarray]] = []
    for route_index, route_row in enumerate(routes):
        points_xy = np.asarray(route_row["points_xy"], dtype=np.float64)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2 or len(points_xy) < 2:
            raise ValueError(f"invalid continuous pedestrian route: {route_row.get('id')}")
        points = np.column_stack(
            [points_xy, np.full(len(points_xy), ground_z, dtype=np.float64)]
        )
        route_centerlines.append((str(route_row["id"]), points_xy))
        models = list(route_row.get("character_model_pool", []))
        if not models:
            raise ValueError(f"route has no character model pool: {route_row.get('id')}")
        for slot in range(slots_per_route):
            initially_active = slot < len(initial_fractions)
            fraction = initial_fractions[slot] if initially_active else 0.0
            spawn, waypoint = _point_at_route_fraction(points, fraction)
            model = Path(models[slot % len(models)])
            expanded.append(
                {
                    "name": f"Route{route_index + 1:02d}_Pedestrian_{slot:02d}",
                    "route_id": str(route_row["id"]),
                    "slot_index": slot,
                    "character_usd": str(assets_root / "Characters" / model),
                    "spawn_xyz": spawn.tolist(),
                    "route_start_xyz": points[0].tolist(),
                    "route_points_xyz": points[1:].tolist(),
                    "preferred_speed_mps": float(
                        route_row.get("preferred_speed_mps", inventory.get("common_speed_mps", 1.0))
                    ),
                    "initial_active": initially_active,
                    "initial_route_arc_fraction": fraction,
                    "initial_waypoint": waypoint,
                }
            )
    payload["pedestrians"] = expanded
    payload["assets_root"] = str(assets_root)
    payload["route_inventory"] = str(inventory_path)
    payload["route_inventory_payload"] = inventory
    pair_clearances: dict[str, float] = {}
    for index, (route_id_a, points_a) in enumerate(route_centerlines):
        for route_id_b, points_b in route_centerlines[index + 1 :]:
            minimum = math.inf
            for start in range(0, len(points_a), 128):
                distances = np.linalg.norm(
                    points_a[start : start + 128, None, :] - points_b[None, :, :],
                    axis=2,
                )
                minimum = min(minimum, float(np.min(distances)))
            pair_clearances[f"{route_id_a}__{route_id_b}"] = minimum
    payload["route_pair_minimum_centerline_distances_m"] = pair_clearances


def load_people_config(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version == 2:
        _expand_continuous_route_config(path, payload)
    elif schema_version != 1:
        raise ValueError(f"unsupported pedestrian config schema: {payload.get('schema_version')}")
    rows = payload.get("pedestrians", [])
    if not rows:
        raise ValueError("pedestrian config contains no pedestrians")
    for row in rows:
        asset = Path(row["character_usd"])
        row["character_usd"] = str(
            (asset if asset.is_absolute() else path.parent / asset).resolve()
        )
        if not Path(row["character_usd"]).is_file():
            raise FileNotFoundError(f"pedestrian asset does not exist: {row['character_usd']}")
        spawn = np.asarray(row["spawn_xyz"], dtype=np.float64)
        route = np.asarray(row["route_points_xyz"], dtype=np.float64)
        if spawn.shape != (3,) or route.ndim != 2 or route.shape[1] != 3 or len(route) < 1:
            raise ValueError(f"invalid pedestrian route: {row.get('name')}")
        if "route_start_xyz" in row:
            route_start = np.asarray(row["route_start_xyz"], dtype=np.float64)
            if route_start.shape != (3,):
                raise ValueError(
                    f"invalid pedestrian route_start_xyz: {row.get('name')}"
                )
    clearance_review = payload.get("static_obstacle_clearance_review")
    if clearance_review is not None:
        reviewed_aabb = np.asarray(
            clearance_review["reviewed_centerline_aabb_xy"], dtype=np.float64
        )
        if reviewed_aabb.shape != (4,) or not (
            reviewed_aabb[0] <= reviewed_aabb[2]
            and reviewed_aabb[1] <= reviewed_aabb[3]
        ):
            raise ValueError("invalid reviewed pedestrian centerline AABB")
        clearance_keys = {
            "required_minimum_body_clearance_m",
            "measured_minimum_body_clearance_m",
        }
        supplied_clearance_keys = clearance_keys.intersection(clearance_review)
        if supplied_clearance_keys and supplied_clearance_keys != clearance_keys:
            raise ValueError(
                "reviewed pedestrian static-obstacle clearance requires both measured and required values"
            )
        if supplied_clearance_keys:
            required = float(clearance_review["required_minimum_body_clearance_m"])
            measured = float(clearance_review["measured_minimum_body_clearance_m"])
            if required < 0.0 or measured < required:
                raise ValueError(
                    "reviewed pedestrian static-obstacle clearance does not meet the requirement"
                )
        for row in rows:
            reviewed_points = np.asarray(
                [
                    row.get("route_start_xyz", row["spawn_xyz"]),
                    row["spawn_xyz"],
                    *row["route_points_xyz"],
                ],
                dtype=np.float64,
            )[:, :2]
            inside = (
                (reviewed_points[:, 0] >= reviewed_aabb[0] - 1.0e-9)
                & (reviewed_points[:, 0] <= reviewed_aabb[2] + 1.0e-9)
                & (reviewed_points[:, 1] >= reviewed_aabb[1] - 1.0e-9)
                & (reviewed_points[:, 1] <= reviewed_aabb[3] + 1.0e-9)
            )
            if not bool(np.all(inside)):
                raise ValueError(
                    f"pedestrian route leaves its reviewed static-clearance area: {row.get('name')}"
                )
    assets_root = Path(payload["assets_root"])
    payload["assets_root"] = str(
        (assets_root if assets_root.is_absolute() else path.parent / assets_root).resolve()
    )
    walk_animation = (
        Path(payload["assets_root"]) / "Animations/stand_walk_loop_in_place.skelanim.usd"
    )
    if not walk_animation.is_file():
        raise FileNotFoundError(f"Isaac People walk animation is missing: {walk_animation}")
    retarget_source_character = (
        Path(payload["assets_root"]) / "Characters/Biped_Setup.usd"
    )
    if not retarget_source_character.is_file():
        raise FileNotFoundError(
            "Isaac People retarget source skeleton is missing: "
            f"{retarget_source_character}"
        )
    payload["config_path"] = str(path)
    payload["walk_animation_usd"] = str(walk_animation)
    payload["retarget_source_character_usd"] = str(retarget_source_character)
    return payload


def author_official_people(
    wrapper_path: Path, config_path: Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Compose character skins and animation targets before ``gym.make``."""
    return author_people_payload(wrapper_path, load_people_config(config_path))


def author_people_payload(
    wrapper_path: Path, payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Compose a validated in-memory People payload before ``gym.make``."""
    from pxr import Gf, Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.Open(str(wrapper_path))
    if stage is None:
        raise RuntimeError(f"could not open wrapper for pedestrians: {wrapper_path}")
    stage.SetEditTarget(stage.GetRootLayer())
    relative_root = "/UrbanVerseAsset/Characters"
    UsdGeom.Xform.Define(stage, relative_root)
    source_path = relative_root + "/WalkAnimationSource"
    source = stage.DefinePrim(source_path)
    source.GetReferences().AddReference(payload["walk_animation_usd"])
    retarget_source_path = relative_root + "/WalkRetargetSource"
    retarget_source = stage.DefinePrim(retarget_source_path)
    retarget_source.GetReferences().AddReference(
        payload["retarget_source_character_usd"]
    )

    parent_translation = np.asarray(
        payload.get("asset_parent_world_translation_xyz", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if parent_translation.shape != (3,):
        raise ValueError("asset_parent_world_translation_xyz must contain three values")
    configured: list[dict[str, str]] = []
    for row in payload["pedestrians"]:
        local_path = f"{relative_root}/{row['name']}"
        prim = UsdGeom.Xform.Define(stage, local_path).GetPrim()
        prim.GetReferences().AddReference(row["character_usd"])
        UsdSkel.Animation.Define(stage, local_path + "/RuntimeWalk")
        transform = UsdGeom.XformCommonAPI(prim)
        local_spawn = np.asarray(row["spawn_xyz"], dtype=np.float64) - parent_translation
        transform.SetTranslate(Gf.Vec3d(*map(float, local_spawn)))
        heading = float(row.get("spawn_heading_deg", 0.0))
        rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr:
            rotate_attr.Set(Gf.Vec3d(0.0, 0.0, heading))
        else:
            transform.SetRotate(
                Gf.Vec3f(0.0, 0.0, heading),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )
        configured.append(
            {
                "name": str(row["name"]),
                "controlled_prim": f"/World/ground/terrain/Characters/{row['name']}",
                "character_usd": str(row["character_usd"]),
                "role": "retargeted_usdskel_animated_pedestrian",
            }
        )
    stage.GetRootLayer().Save()
    return payload, configured


def configure_preauthored_people(cfg: Any, configured: list[dict[str, str]]) -> None:
    """Register precomposed people as global Isaac Lab/Fabric assets."""
    from isaaclab.assets import AssetBaseCfg

    for index, row in enumerate(configured):
        setattr(
            cfg.scene,
            f"animated_pedestrian_{index:02d}",
            AssetBaseCfg(prim_path=row["controlled_prim"], spawn=None, collision_group=-1),
        )


class OfficialPeopleManager:
    """Drive NVIDIA People skins with UsdSkel and reviewed sidewalk routes.

    The compact Isaac 4.5 experience does not initialize
    ``omni.anim.graph.core`` reliably.  We therefore reuse NVIDIA's skinned
    People meshes and an officially retargeted in-place walk clip, while root
    motion and conservative circle-based yielding stay explicit and testable
    here.
    """

    def __init__(self, stage: Any, config: dict[str, Any], run_dir: Path) -> None:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        if not extension_manager.is_extension_enabled("omni.anim.retarget.core"):
            extension_manager.set_extension_enabled_immediate(
                "omni.anim.retarget.core", True
            )
        from omni.anim.retarget.core import RetargetController
        from omni.anim.retarget.core.scripts.utils import (
            convert_matrix_to_trans_rots,
            convert_trans_rots_to_pxr_matrices,
        )
        from pxr import Usd, UsdGeom, UsdSkel

        self.stage = stage
        self.config = config
        self.rows = list(config["pedestrians"])
        self.character_root_prim = str(config["character_root_prim"])
        self.dt = float(config.get("simulation_dt_s", 0.02))
        self._convert_matrix_to_trans_rots = convert_matrix_to_trans_rots
        self._convert_trans_rots_to_pxr_matrices = convert_trans_rots_to_pxr_matrices
        self.command_path = Path(run_dir) / "metadata/pedestrian_commands.txt"
        self.command_path.write_text(
            "\n".join(
                f"{row['name']} GoTo {float(point[0]):.6f} {float(point[1]):.6f} "
                f"{float(point[2]):.6f}"
                for row in self.rows
                for point in row["route_points_xyz"]
            )
            + "\n",
            encoding="utf-8",
        )

        roots = [
            stage.GetPrimAtPath(f"{self.character_root_prim}/{row['name']}")
            for row in self.rows
        ]
        if any(not prim.IsValid() for prim in roots):
            missing = [row["name"] for row, prim in zip(self.rows, roots) if not prim.IsValid()]
            raise RuntimeError(f"preauthored pedestrian roots are missing: {missing}")
        self.roots = roots
        self.skelroots = []
        self.skeletons = []
        for root in roots:
            skelroot_candidates = [
                prim for prim in Usd.PrimRange(root) if prim.IsA(UsdSkel.Root)
            ]
            skeleton_candidates = [
                prim for prim in Usd.PrimRange(root) if prim.IsA(UsdSkel.Skeleton)
            ]
            if not skelroot_candidates:
                raise RuntimeError(f"People asset contains no SkelRoot: {root.GetPath()}")
            if not skeleton_candidates:
                raise RuntimeError(f"People asset contains no Skeleton: {root.GetPath()}")
            self.skelroots.append(skelroot_candidates[0])
            self.skeletons.append(UsdSkel.Skeleton(skeleton_candidates[0]))

        source_root = stage.GetPrimAtPath(f"{self.character_root_prim}/WalkAnimationSource")
        source_candidates = [
            prim for prim in Usd.PrimRange(source_root) if prim.IsA(UsdSkel.Animation)
        ]
        if not source_candidates:
            raise RuntimeError("direct People walk animation source is missing")
        self.source_animation = UsdSkel.Animation(source_candidates[0])
        self.source_attrs = [
            self.source_animation.GetTranslationsAttr(),
            self.source_animation.GetRotationsAttr(),
            self.source_animation.GetScalesAttr(),
        ]
        sample_times = sorted(
            {
                float(value)
                for attr in self.source_attrs
                for value in attr.GetTimeSamples()
            }
        )
        if len(sample_times) < 2:
            raise RuntimeError("People walk animation contains no usable time samples")
        self.animation_start_time_code = sample_times[0]
        self.animation_period_time_codes = sample_times[-1] - sample_times[0]
        self.animation_rate_time_codes_per_s = float(
            config.get("animation_rate_time_codes_per_s", 30.0)
        )
        self.animation_update_hz = float(config.get("animation_update_hz", 25.0))
        self.animation_stop_threshold_mps = float(
            config.get("animation_stop_threshold_mps", 0.03)
        )
        if self.animation_stop_threshold_mps < 0.0:
            raise ValueError("animation_stop_threshold_mps must be nonnegative")
        self.animation_update_stride = max(
            1, int(round(1.0 / max(self.dt * self.animation_update_hz, 1.0e-9)))
        )

        retarget_source_root = stage.GetPrimAtPath(
            f"{self.character_root_prim}/WalkRetargetSource"
        )
        retarget_source_candidates = [
            prim
            for prim in Usd.PrimRange(retarget_source_root)
            if prim.IsA(UsdSkel.Skeleton)
        ]
        if not retarget_source_candidates:
            raise RuntimeError("People retarget source Skeleton is missing")
        self.retarget_source_skeleton = UsdSkel.Skeleton(
            retarget_source_candidates[0]
        )

        self.target_animations = []
        self.retarget_controllers = []
        for root, skeleton in zip(roots, self.skeletons):
            target = UsdSkel.Animation(stage.GetPrimAtPath(f"{root.GetPath()}/RuntimeWalk"))
            if not target:
                raise RuntimeError(f"runtime walk target is missing below {root.GetPath()}")
            joints = skeleton.GetJointsAttr().Get()
            if joints is not None:
                target.CreateJointsAttr().Set(joints)
            UsdSkel.BindingAPI.Apply(
                skeleton.GetPrim()
            ).CreateAnimationSourceRel().SetTargets(
                [target.GetPrim().GetPath()]
            )
            self.target_animations.append(target)
            self.retarget_controllers.append(
                RetargetController(
                    None,
                    str(self.retarget_source_skeleton.GetPath()),
                    -1,
                    str(skeleton.GetPath()),
                )
            )

        self.pose_records = []
        for root in roots:
            xformable = UsdGeom.Xformable(root)
            parent_xform = UsdGeom.Xformable(root.GetParent())
            parent_world = (
                np.asarray(parent_xform.ComputeLocalToWorldTransform(0), dtype=np.float64).T
                if parent_xform
                else np.eye(4, dtype=np.float64)
            )
            xformable.ClearXformOpOrder()
            op = xformable.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble, "pedestrianRuntime"
            )
            self.pose_records.append({"parent_world": parent_world, "op": op})

        self.lifecycle_mode = str(config.get("lifecycle_mode", "ping_pong"))
        self.continuous_spawn_despawn = (
            self.lifecycle_mode == "spawn_at_start_despawn_at_goal"
        )
        self.rng = np.random.default_rng(int(config.get("random_seed", 20260822)))
        self.minimum_spawn_headway_m = float(
            config.get("minimum_spawn_headway_m", 6.0)
        )
        self.maximum_turn_rate_deg_s = float(
            config.get("maximum_turn_rate_deg_s", 180.0)
        )
        spawn_interval = config.get(
            "spawn_interval_s", {"minimum": 8.0, "maximum": 14.0}
        )
        self.spawn_interval_min_s = float(spawn_interval.get("minimum", 8.0))
        self.spawn_interval_max_s = float(spawn_interval.get("maximum", 14.0))
        self.lifecycle_events: list[dict[str, Any]] = []
        self.route_schedulers: dict[str, dict[str, Any]] = {}
        self.agents: list[dict[str, Any]] = []
        for index, row in enumerate(self.rows):
            route = np.asarray(
                [row.get("route_start_xyz", row["spawn_xyz"]), *row["route_points_xyz"]],
                dtype=np.float64,
            )
            initial_position = np.asarray(row["spawn_xyz"], dtype=np.float64)
            initial_waypoint = int(row.get("initial_waypoint", 1))
            initial_delta = route[initial_waypoint, :2] - initial_position[:2]
            if np.linalg.norm(initial_delta) < 1.0e-9:
                initial_delta = route[1, :2] - route[0, :2]
            heading = math.atan2(initial_delta[1], initial_delta[0])
            self.agents.append(
                {
                    "name": str(row["name"]),
                    "position": initial_position.copy(),
                    "route": route,
                    "waypoint": initial_waypoint,
                    "direction": 1,
                    "active": bool(row.get("initial_active", True)),
                    "route_id": str(row.get("route_id", row["name"])),
                    "slot_index": int(row.get("slot_index", index)),
                    "preferred_speed": float(row.get("preferred_speed_mps", 1.15)),
                    "speed": 0.0,
                    "heading": heading,
                    "endpoint_reversal_count": 0,
                    "spawn_count": 1 if bool(row.get("initial_active", True)) else 0,
                    "completion_count": 0,
                    "waypoint_transition_count": 0,
                    "phase_offset": 0.23 * index,
                    "animation_time_s": (
                        0.23
                        * index
                        * self.animation_period_time_codes
                        / self.animation_rate_time_codes_per_s
                    ),
                }
            )
            route_id = str(row.get("route_id", row["name"]))
            self.route_schedulers.setdefault(
                route_id,
                {
                    "next_spawn_time_s": float(
                        self.rng.uniform(self.spawn_interval_min_s, self.spawn_interval_max_s)
                    ),
                    "spawn_count": 0,
                    "completion_count": 0,
                    "deferred_spawn_count": 0,
                },
            )
            if bool(row.get("initial_active", True)):
                self.route_schedulers[route_id]["spawn_count"] += 1
        self.path_lengths = {str(row["name"]): 0.0 for row in self.rows}
        self.maximum_simultaneously_moving = 0
        self.update_count = 0
        self.simulation_time_s = 0.0
        self.minimum_vehicle_clearance_m = math.inf
        self.minimum_go2_clearance_m = math.inf
        self.minimum_pedestrian_clearance_m = math.inf
        self.yield_step_count = 0
        self.maximum_heading_motion_error_deg = 0.0
        self.maximum_turn_angle_deg = 0.0
        self.maximum_active_pedestrian_count = 0
        self.animation_paused_steps = {agent["name"]: 0 for agent in self.agents}
        self.animation_advanced_steps = {agent["name"]: 0 for agent in self.agents}
        for index, agent in enumerate(self.agents):
            self._set_active(index, bool(agent["active"]), author_pose=True)

    def _set_visibility(self, index: int, visible: bool) -> None:
        from pxr import UsdGeom

        imageable = UsdGeom.Imageable(self.roots[index])
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _set_active(self, index: int, active: bool, *, author_pose: bool) -> None:
        agent = self.agents[index]
        agent["active"] = bool(active)
        self._set_visibility(index, bool(active))
        if active and author_pose:
            self._author_pose(index, agent["position"], agent["heading"])
            self._apply_animation_sample(index)

    def _schedule_next_spawn(self, route_id: str) -> None:
        scheduler = self.route_schedulers[route_id]
        scheduler["next_spawn_time_s"] = self.simulation_time_s + float(
            self.rng.uniform(self.spawn_interval_min_s, self.spawn_interval_max_s)
        )

    def _spawn_due_agents(self) -> None:
        if not self.continuous_spawn_despawn:
            return
        for route_id, scheduler in self.route_schedulers.items():
            if self.simulation_time_s + 1.0e-9 < scheduler["next_spawn_time_s"]:
                continue
            candidates = [
                (index, agent)
                for index, agent in enumerate(self.agents)
                if agent["route_id"] == route_id and not agent["active"]
            ]
            active = [
                agent
                for agent in self.agents
                if agent["route_id"] == route_id and agent["active"]
            ]
            start = candidates[0][1]["route"][0][:2] if candidates else None
            headway_ok = bool(
                start is not None
                and all(
                    float(np.linalg.norm(agent["position"][:2] - start))
                    >= self.minimum_spawn_headway_m
                    for agent in active
                )
            )
            if not candidates or not headway_ok:
                scheduler["deferred_spawn_count"] += 1
                scheduler["next_spawn_time_s"] = self.simulation_time_s + 0.5
                continue
            index, agent = min(candidates, key=lambda item: item[1]["spawn_count"])
            agent["position"] = agent["route"][0].copy()
            agent["waypoint"] = 1
            agent["direction"] = 1
            delta = agent["route"][1, :2] - agent["route"][0, :2]
            agent["heading"] = math.atan2(delta[1], delta[0])
            agent["speed"] = 0.0
            agent["spawn_count"] += 1
            scheduler["spawn_count"] += 1
            self.lifecycle_events.append(
                {
                    "time_s": self.simulation_time_s,
                    "event": "spawn",
                    "route_id": route_id,
                    "agent": agent["name"],
                }
            )
            self._set_active(index, True, author_pose=True)
            self._schedule_next_spawn(route_id)

    def _complete_agent(self, index: int) -> None:
        agent = self.agents[index]
        agent["completion_count"] += 1
        scheduler = self.route_schedulers[agent["route_id"]]
        scheduler["completion_count"] += 1
        self.lifecycle_events.append(
            {
                "time_s": self.simulation_time_s,
                "event": "despawn_at_goal",
                "route_id": agent["route_id"],
                "agent": agent["name"],
            }
        )
        agent["speed"] = 0.0
        self._set_active(index, False, author_pose=False)

    def _author_pose(self, index: int, position: np.ndarray, heading: float) -> None:
        from pxr import Gf

        visual_yaw = people_visual_yaw_from_route_heading(heading)
        cosine, sine = math.cos(visual_yaw), math.sin(visual_yaw)
        world = np.eye(4, dtype=np.float64)
        world[:3, :3] = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        world[:3, 3] = np.asarray(position, dtype=np.float64)
        record = self.pose_records[index]
        local = np.linalg.inv(record["parent_world"]) @ world
        usd_matrix = local.T
        authored = Gf.Matrix4d(1.0)
        for row in range(4):
            authored.SetRow(row, Gf.Vec4d(*usd_matrix[row].tolist()))
        record["op"].Set(authored)

    def _apply_animation_sample(self, index: int) -> None:
        from pxr import Usd, Vt

        agent = self.agents[index]
        phase_time = (
            agent["animation_time_s"] * self.animation_rate_time_codes_per_s
        ) % self.animation_period_time_codes
        time_code = Usd.TimeCode(self.animation_start_time_code + phase_time)
        source_transforms = self.source_animation.GetTransforms(time_code)
        source_translations, source_rotations = self._convert_matrix_to_trans_rots(
            source_transforms
        )
        target_translations, target_rotations = self.retarget_controllers[index].retarget(
            source_translations, source_rotations
        )
        if len(target_translations) != len(self.skeletons[index].GetJointsAttr().Get()):
            raise RuntimeError(
                "People animation retargeting returned an invalid target pose for "
                f"{self.agents[index]['name']}"
            )
        target_transforms = self._convert_trans_rots_to_pxr_matrices(
            target_translations, target_rotations
        )
        self.target_animations[index].SetTransforms(
            Vt.Matrix4dArray(target_transforms), Usd.TimeCode.Default()
        )

    def _advance_animation_clock(self, index: int) -> None:
        agent = self.agents[index]
        previous = float(agent["animation_time_s"])
        current = advance_walk_animation_clock(
            previous,
            dt_s=self.dt,
            actual_speed_mps=float(agent["speed"]),
            preferred_speed_mps=float(agent["preferred_speed"]),
            stop_threshold_mps=self.animation_stop_threshold_mps,
        )
        agent["animation_time_s"] = current
        counter = (
            self.animation_advanced_steps
            if current > previous
            else self.animation_paused_steps
        )
        counter[agent["name"]] += 1

    def prepare_step(
        self,
        traffic_agents: list[dict[str, Any]] | None,
        go2_xy: np.ndarray,
        go2_velocity_xy: np.ndarray,
    ) -> None:
        """Advance sidewalk routes while yielding to car, Go2, and people circles."""
        del go2_velocity_xy
        active_vehicles = [
            item
            for item in (traffic_agents or [])
            if item.get("status") in ("moving", "stopped")
        ]
        go2_xy = np.asarray(go2_xy, dtype=np.float64)
        self._spawn_due_agents()
        moving_count = 0
        for index, agent in enumerate(self.agents):
            if not agent["active"]:
                continue
            route = agent["route"]
            target = route[agent["waypoint"]]
            delta = target[:2] - agent["position"][:2]
            distance_to_target = float(np.linalg.norm(delta))
            if distance_to_target < 0.18:
                if agent["direction"] > 0 and agent["waypoint"] == len(route) - 1:
                    if self.continuous_spawn_despawn:
                        self._complete_agent(index)
                        continue
                    else:
                        agent["direction"] = -1
                        agent["endpoint_reversal_count"] += 1
                elif agent["direction"] < 0 and agent["waypoint"] == 0:
                    agent["direction"] = 1
                    agent["endpoint_reversal_count"] += 1
                agent["waypoint"] += agent["direction"]
                agent["waypoint_transition_count"] += 1
                target = route[agent["waypoint"]]
                delta = target[:2] - agent["position"][:2]
                distance_to_target = float(np.linalg.norm(delta))
            direction = delta / max(distance_to_target, 1.0e-9)
            desired_heading = math.atan2(direction[1], direction[0])
            heading_error = math.atan2(
                math.sin(desired_heading - agent["heading"]),
                math.cos(desired_heading - agent["heading"]),
            )
            maximum_heading_step = math.radians(self.maximum_turn_rate_deg_s) * self.dt
            applied_heading_step = float(
                np.clip(heading_error, -maximum_heading_step, maximum_heading_step)
            )
            agent["heading"] += applied_heading_step
            if abs(heading_error) <= maximum_heading_step:
                agent["heading"] = desired_heading
            self.maximum_turn_angle_deg = max(
                self.maximum_turn_angle_deg, abs(math.degrees(applied_heading_step))
            )
            heading_aligned = abs(heading_error) <= maximum_heading_step
            speed_scale = 1.0
            for vehicle in active_vehicles:
                relative = np.asarray(vehicle["position"], dtype=np.float64) - agent["position"][:2]
                center_distance = float(np.linalg.norm(relative))
                radius = float(vehicle.get("radius", 1.5)) + 0.42
                clearance = center_distance - radius
                self.minimum_vehicle_clearance_m = min(self.minimum_vehicle_clearance_m, clearance)
                lateral = abs(direction[0] * relative[1] - direction[1] * relative[0])
                if float(np.dot(relative, direction)) > 0.0 and lateral < radius + 0.35:
                    speed_scale = min(
                        speed_scale, float(np.clip((clearance - 0.25) / 1.5, 0.0, 1.0))
                    )
            go2_relative = go2_xy - agent["position"][:2]
            go2_clearance = float(np.linalg.norm(go2_relative)) - 0.97
            self.minimum_go2_clearance_m = min(self.minimum_go2_clearance_m, go2_clearance)
            go2_lateral = abs(
                direction[0] * go2_relative[1] - direction[1] * go2_relative[0]
            )
            if float(np.dot(go2_relative, direction)) > 0.0 and go2_lateral < 1.15:
                speed_scale = min(
                    speed_scale, float(np.clip((go2_clearance - 0.2) / 1.2, 0.0, 1.0))
                )
            for other_index, other in enumerate(self.agents):
                if other_index == index or not other["active"]:
                    continue
                relative = other["position"][:2] - agent["position"][:2]
                clearance = float(np.linalg.norm(relative)) - 0.84
                self.minimum_pedestrian_clearance_m = min(
                    self.minimum_pedestrian_clearance_m, clearance
                )
                lateral = abs(direction[0] * relative[1] - direction[1] * relative[0])
                if float(np.dot(relative, direction)) > 0.0 and lateral < 1.0:
                    speed_scale = min(
                        speed_scale, float(np.clip((clearance - 0.2) / 1.0, 0.0, 1.0))
                    )
            speed = float(agent["preferred_speed"] * speed_scale) if heading_aligned else 0.0
            previous_position = agent["position"][:2].copy()
            step_distance = min(distance_to_target, speed * self.dt)
            agent["position"][:2] += direction * step_distance
            agent["speed"] = speed
            if speed > 0.02:
                motion = agent["position"][:2] - previous_position
                if float(np.linalg.norm(motion)) > 1.0e-9:
                    motion_heading = math.atan2(motion[1], motion[0])
                    error = math.atan2(
                        math.sin(agent["heading"] - motion_heading),
                        math.cos(agent["heading"] - motion_heading),
                    )
                    self.maximum_heading_motion_error_deg = max(
                        self.maximum_heading_motion_error_deg,
                        abs(math.degrees(error)),
                    )
                moving_count += 1
            if speed_scale < 0.999:
                self.yield_step_count += 1
            self.path_lengths[agent["name"]] += step_distance
            self._author_pose(index, agent["position"], agent["heading"])
            self._advance_animation_clock(index)
            if self.update_count % self.animation_update_stride == 0:
                self._apply_animation_sample(index)
        self.maximum_simultaneously_moving = max(self.maximum_simultaneously_moving, moving_count)
        active_count = sum(bool(agent["active"]) for agent in self.agents)
        self.maximum_active_pedestrian_count = max(
            self.maximum_active_pedestrian_count, active_count
        )
        self.simulation_time_s += self.dt

    def sample(self) -> None:
        self.update_count += 1

    def refresh_visual_transforms(self) -> None:
        for index, agent in enumerate(self.agents):
            if not agent["active"]:
                continue
            self._author_pose(index, agent["position"], agent["heading"])
            self._apply_animation_sample(index)

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    def _visual_bboxes(self) -> list[dict[str, Any]]:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        rows = []
        for config_row in self.rows:
            prim = self.stage.GetPrimAtPath(
                f"{self.character_root_prim}/{config_row['name']}"
            )
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum = np.asarray(bounds.GetMin(), dtype=np.float64)
            maximum = np.asarray(bounds.GetMax(), dtype=np.float64)
            rows.append(
                {
                    "name": str(config_row["name"]),
                    "minimum_xyz": minimum.tolist(),
                    "maximum_xyz": maximum.tolist(),
                    "size_xyz": (maximum - minimum).tolist(),
                    "finite": bool(np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum))),
                }
            )
        return rows

    def summary(self) -> dict[str, Any]:
        return {
            "implementation": (
                "Isaac Sim 4.5 People skins + omni.anim.retarget.core looping "
                "UsdSkel walk clip + waypoint circle yielding"
            ),
            "pedestrian_count": len(self.rows),
            "lifecycle_mode": self.lifecycle_mode,
            "continuous_spawn_despawn": self.continuous_spawn_despawn,
            "active_pedestrian_count": sum(
                bool(agent["active"]) for agent in self.agents
            ),
            "maximum_active_pedestrian_count": self.maximum_active_pedestrian_count,
            "asset_root": self.config["assets_root"],
            "command_file": str(self.command_path),
            "navmesh_enabled": False,
            "dynamic_avoidance_enabled": bool(self.config.get("dynamic_avoidance_enabled", True)),
            "maximum_simultaneously_moving_pedestrian_count": self.maximum_simultaneously_moving,
            "path_lengths_m": self.path_lengths,
            "walk_animation_usd": self.config["walk_animation_usd"],
            "retarget_source_character_usd": self.config[
                "retarget_source_character_usd"
            ],
            "character_local_forward_axis": "-Y",
            "character_up_axis": "+Z",
            "route_heading_alignment": "visual_yaw_rad = pi/2 + route_heading_rad",
            "retarget_source_joint_count": len(
                self.retarget_source_skeleton.GetJointsAttr().Get()
            ),
            "target_joint_counts": [
                len(skeleton.GetJointsAttr().Get()) for skeleton in self.skeletons
            ],
            "root_motion_controller": "kinematic reviewed sidewalk waypoint routes",
            "animation_time_code_range": [
                self.animation_start_time_code,
                self.animation_start_time_code + self.animation_period_time_codes,
            ],
            "minimum_vehicle_clearance_m": self._finite_or_none(
                self.minimum_vehicle_clearance_m
            ),
            "minimum_go2_clearance_m": self._finite_or_none(self.minimum_go2_clearance_m),
            "minimum_pedestrian_clearance_m": self._finite_or_none(
                self.minimum_pedestrian_clearance_m
            ),
            "static_obstacle_clearance_review": self.config.get(
                "static_obstacle_clearance_review"
            ),
            "yield_step_count": self.yield_step_count,
            "maximum_heading_motion_error_deg": self.maximum_heading_motion_error_deg,
            "maximum_turn_angle_deg": self.maximum_turn_angle_deg,
            "maximum_turn_rate_deg_s": self.maximum_turn_rate_deg_s,
            "animation_update_hz": self.animation_update_hz,
            "animation_clock_mode": "actual root speed scaled; frozen at stop",
            "animation_stop_threshold_mps": self.animation_stop_threshold_mps,
            "animation_paused_steps": self.animation_paused_steps,
            "animation_advanced_steps": self.animation_advanced_steps,
            "route_schedulers": self.route_schedulers,
            "route_pair_minimum_centerline_distances_m": self.config.get(
                "route_pair_minimum_centerline_distances_m", {}
            ),
            "total_spawn_count": sum(
                int(agent["spawn_count"]) for agent in self.agents
            ),
            "total_completion_count": sum(
                int(agent["completion_count"]) for agent in self.agents
            ),
            "spawn_counts": {
                agent["name"]: int(agent["spawn_count"]) for agent in self.agents
            },
            "completion_counts": {
                agent["name"]: int(agent["completion_count"]) for agent in self.agents
            },
            "waypoint_transition_counts": {
                agent["name"]: int(agent["waypoint_transition_count"])
                for agent in self.agents
            },
            "lifecycle_events": self.lifecycle_events,
            "final_positions_xyz": {
                agent["name"]: agent["position"].tolist() for agent in self.agents
            },
            "endpoint_reversal_counts": {
                agent["name"]: int(agent["endpoint_reversal_count"])
                for agent in self.agents
            },
            "final_route_headings_deg": {
                agent["name"]: float(math.degrees(agent["heading"]))
                for agent in self.agents
            },
            "final_visual_yaws_deg": {
                agent["name"]: float(
                    math.degrees(people_visual_yaw_from_route_heading(agent["heading"]))
                )
                for agent in self.agents
            },
            "visual_bboxes": self._visual_bboxes(),
            "update_count": self.update_count,
        }
