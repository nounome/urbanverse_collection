#!/usr/bin/env python3
"""Phase-one real-vehicle crossing and Go2-owned yielding primitives.

The vehicle follows a timestamp-only schedule and never observes the robot.
The Go2-side controller may scale a route follower's velocity command to yield.
USD imports are intentionally local so the analytic controller is unit-testable
without launching Isaac Sim.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def unit_xy(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(2)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-9:
        raise ValueError("zero-length XY direction")
    return vector / length


def point_to_obb_clearance(
    point_xy: np.ndarray,
    center_xy: np.ndarray,
    heading_rad: float,
    half_extents_xy: np.ndarray,
    point_radius: float,
) -> float:
    """Signed clearance from a disc to a 2-D oriented rectangle."""

    relative = np.asarray(point_xy, dtype=np.float64) - np.asarray(center_xy, dtype=np.float64)
    cosine = math.cos(heading_rad)
    sine = math.sin(heading_rad)
    local = np.asarray(
        [cosine * relative[0] + sine * relative[1], -sine * relative[0] + cosine * relative[1]],
        dtype=np.float64,
    )
    outside = np.maximum(np.abs(local) - np.asarray(half_extents_xy, dtype=np.float64), 0.0)
    outside_distance = float(np.linalg.norm(outside))
    inside_depth = min(
        float(np.asarray(half_extents_xy)[0] - abs(local[0])),
        float(np.asarray(half_extents_xy)[1] - abs(local[1])),
    )
    signed_point_distance = outside_distance if outside_distance > 0.0 else -inside_depth
    return signed_point_distance - float(point_radius)


@dataclass(frozen=True)
class CrossingSchedule:
    conflict_xy: np.ndarray
    direction_xy: np.ndarray
    distance: float = 12.0
    crossing_time_s: float = 6.0
    speed: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflict_xy", np.asarray(self.conflict_xy, dtype=np.float64).reshape(2))
        object.__setattr__(self, "direction_xy", unit_xy(self.direction_xy))
        if self.distance <= 0.0 or self.speed <= 0.0:
            raise ValueError("distance and speed must be positive")
        if self.crossing_time_s < self.travel_duration_s / 2.0:
            raise ValueError("crossing time places vehicle motion before t=0")

    @property
    def travel_duration_s(self) -> float:
        return float(self.distance / self.speed)

    @property
    def start_time_s(self) -> float:
        return float(self.crossing_time_s - self.travel_duration_s / 2.0)

    @property
    def end_time_s(self) -> float:
        return float(self.crossing_time_s + self.travel_duration_s / 2.0)

    @property
    def start_xy(self) -> np.ndarray:
        return self.conflict_xy - self.direction_xy * (self.distance / 2.0)

    @property
    def end_xy(self) -> np.ndarray:
        return self.conflict_xy + self.direction_xy * (self.distance / 2.0)

    @property
    def heading_rad(self) -> float:
        return math.atan2(float(self.direction_xy[1]), float(self.direction_xy[0]))

    def sample(self, timestamp_s: float) -> dict[str, Any]:
        raw_progress = (float(timestamp_s) - self.start_time_s) / self.travel_duration_s
        progress = float(np.clip(raw_progress, 0.0, 1.0))
        moving = 0.0 < raw_progress < 1.0
        center = self.start_xy + self.direction_xy * (self.distance * progress)
        velocity = self.direction_xy * (self.speed if moving else 0.0)
        return {
            "timestamp_s": float(timestamp_s),
            "progress": progress,
            "moving": moving,
            "completed": raw_progress >= 1.0,
            "center_xy": center,
            "velocity_xy": velocity,
            "heading_rad": self.heading_rad,
            "conflict_xy": self.conflict_xy,
            "crossing_time_s": self.crossing_time_s,
        }


class Go2YieldController:
    """Scale route commands so Go2 yields to a non-reactive moving vehicle."""

    def __init__(
        self,
        *,
        enabled: bool,
        dt: float,
        go2_radius: float = 0.45,
        time_horizon_s: float = 3.5,
        predicted_clearance_trigger: float = 0.40,
        emergency_clearance: float = 0.65,
        resume_clearance: float = 1.20,
        deceleration_rate: float = 1.5,
        acceleration_rate: float = 0.65,
    ) -> None:
        self.enabled = bool(enabled)
        self.dt = float(dt)
        self.go2_radius = float(go2_radius)
        self.time_horizon_s = float(time_horizon_s)
        self.predicted_clearance_trigger = float(predicted_clearance_trigger)
        self.emergency_clearance = float(emergency_clearance)
        self.resume_clearance = float(resume_clearance)
        self.deceleration_rate = float(deceleration_rate)
        self.acceleration_rate = float(acceleration_rate)
        self.speed_scale = 1.0
        self.state = "disabled" if not enabled else "tracking"
        self.transitions: list[dict[str, Any]] = []
        self.intervention_steps = 0
        self.full_stop_steps = 0
        self.minimum_clearance = math.inf
        self.minimum_predicted_clearance = math.inf
        self.minimum_positive_ttc = math.inf

    def _transition(self, new_state: str, timestamp_s: float, reason: str) -> None:
        if new_state == self.state:
            return
        self.transitions.append(
            {"timestamp_s": float(timestamp_s), "from": self.state, "to": new_state, "reason": reason}
        )
        self.state = new_state

    def apply(
        self,
        base_command: dict[str, Any],
        *,
        timestamp_s: float,
        go2_position_xy: np.ndarray,
        go2_yaw_rad: float,
        vehicle_state: dict[str, Any],
        vehicle_half_extents_xy: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        command = copy.deepcopy(base_command)
        center = np.asarray(vehicle_state["center_xy"], dtype=np.float64)
        velocity = np.asarray(vehicle_state["velocity_xy"], dtype=np.float64)
        go2_position = np.asarray(go2_position_xy, dtype=np.float64)
        half_extents = np.asarray(vehicle_half_extents_xy, dtype=np.float64)
        clearance = point_to_obb_clearance(
            go2_position,
            center,
            float(vehicle_state["heading_rad"]),
            half_extents,
            self.go2_radius,
        )
        self.minimum_clearance = min(self.minimum_clearance, clearance)

        forward = np.asarray([math.cos(go2_yaw_rad), math.sin(go2_yaw_rad)], dtype=np.float64)
        left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        intended_velocity = (
            forward * float(base_command.get("linear_x", 0.0))
            + left * float(base_command.get("linear_y", 0.0))
        )
        relative_position = center - go2_position
        relative_velocity = velocity - intended_velocity
        relative_speed_squared = float(np.dot(relative_velocity, relative_velocity))
        if relative_speed_squared > 1.0e-9:
            closest_time = float(
                np.clip(
                    -float(np.dot(relative_position, relative_velocity)) / relative_speed_squared,
                    0.0,
                    self.time_horizon_s,
                )
            )
        else:
            closest_time = 0.0
        predicted_separation = float(
            np.linalg.norm(relative_position + relative_velocity * closest_time)
        )
        vehicle_bound_radius = float(np.linalg.norm(half_extents))
        predicted_clearance = predicted_separation - vehicle_bound_radius - self.go2_radius
        self.minimum_predicted_clearance = min(self.minimum_predicted_clearance, predicted_clearance)
        if closest_time > 0.0:
            self.minimum_positive_ttc = min(self.minimum_positive_ttc, closest_time)

        conflict_predicted = (
            closest_time > 0.0
            and closest_time < self.time_horizon_s
            and predicted_clearance < self.predicted_clearance_trigger
            and not bool(vehicle_state.get("completed", False))
        )
        scheduled_time_to_crossing = float(vehicle_state.get("crossing_time_s", timestamp_s) - timestamp_s)
        scheduled_conflict_distance = float(
            np.linalg.norm(np.asarray(vehicle_state.get("conflict_xy", center), dtype=np.float64) - go2_position)
        )
        scheduled_conflict = (
            0.0 < scheduled_time_to_crossing < self.time_horizon_s + 1.0
            and scheduled_conflict_distance < 3.5
            and not bool(vehicle_state.get("completed", False))
        )
        emergency = clearance < self.emergency_clearance
        vehicle_cleared = bool(vehicle_state.get("completed", False)) or (
            float(vehicle_state.get("progress", 0.0)) > 0.64 and clearance > self.resume_clearance
        )

        if self.enabled:
            if self.state in ("tracking", "resuming") and (scheduled_conflict or conflict_predicted or emergency):
                reason = "scheduled crossing" if scheduled_conflict else "predicted conflict" if conflict_predicted else "clearance"
                self._transition("yielding", timestamp_s, reason)
            elif self.state in ("yielding", "stopped") and vehicle_cleared:
                self._transition("resuming", timestamp_s, "vehicle cleared conflict zone")

            target_scale = 1.0
            if self.state in ("yielding", "stopped"):
                target_scale = 0.0
            rate = self.deceleration_rate if target_scale < self.speed_scale else self.acceleration_rate
            delta = float(np.clip(target_scale - self.speed_scale, -rate * self.dt, rate * self.dt))
            self.speed_scale = float(np.clip(self.speed_scale + delta, 0.0, 1.0))
            if self.state == "yielding" and self.speed_scale <= 0.02:
                self.speed_scale = 0.0
                self._transition("stopped", timestamp_s, "commanded full stop")
            if self.state == "resuming" and self.speed_scale >= 0.999:
                self.speed_scale = 1.0
                self._transition("tracking", timestamp_s, "full route speed restored")
        else:
            self.speed_scale = 1.0

        original_linear_x = float(base_command.get("linear_x", 0.0))
        original_linear_y = float(base_command.get("linear_y", 0.0))
        command["linear_x"] = original_linear_x * self.speed_scale
        command["linear_y"] = original_linear_y * self.speed_scale
        command["angular_z"] = float(base_command.get("angular_z", 0.0)) * self.speed_scale
        command["label"] = f"{base_command.get('label', 'route')}|phase1:{self.state}"
        intervened = self.enabled and self.speed_scale < 0.999 and (
            abs(original_linear_x) > 1.0e-4 or abs(original_linear_y) > 1.0e-4
        )
        self.intervention_steps += int(intervened)
        stopped = intervened and self.speed_scale <= 1.0e-6
        self.full_stop_steps += int(stopped)
        diagnostics = {
            "enabled": self.enabled,
            "state": self.state,
            "speed_scale": self.speed_scale,
            "intervened": intervened,
            "full_stop": stopped,
            "current_clearance_m": clearance,
            "closest_approach_time_s": closest_time,
            "predicted_clearance_m": predicted_clearance,
            "conflict_predicted": conflict_predicted,
            "scheduled_conflict": scheduled_conflict,
            "scheduled_time_to_crossing_s": scheduled_time_to_crossing,
            "scheduled_conflict_distance_m": scheduled_conflict_distance,
            "emergency_clearance_triggered": emergency,
            "vehicle_cleared": vehicle_cleared,
            "base_route_command": {
                "linear_x": original_linear_x,
                "linear_y": original_linear_y,
                "angular_z": float(base_command.get("angular_z", 0.0)),
            },
        }
        return command, diagnostics

    def summary(self) -> dict[str, Any]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(value) else None

        return {
            "enabled": self.enabled,
            "responsibility": "Go2-only; the vehicle schedule never reads robot state",
            "final_state": self.state,
            "final_speed_scale": self.speed_scale,
            "intervention_steps": self.intervention_steps,
            "full_stop_steps": self.full_stop_steps,
            "minimum_clearance_m": finite_or_none(self.minimum_clearance),
            "minimum_predicted_clearance_m": finite_or_none(self.minimum_predicted_clearance),
            "minimum_positive_ttc_s": finite_or_none(self.minimum_positive_ttc),
            "transitions": self.transitions,
        }


class OriginalSceneVehicleActor:
    """Move an original scene vehicle root on an isolated session layer."""

    def __init__(self, stage, prim_path: str, schedule: CrossingSchedule) -> None:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        self.stage = stage
        requested_prim_path = prim_path
        self.schedule = schedule
        self.layer = Sdf.Layer.CreateAnonymous("phase1_original_vehicle_motion.usda")
        stage.GetSessionLayer().subLayerPaths.append(self.layer.identifier)
        stage.SetEditTarget(self.layer)
        candidate_paths = [prim_path]
        if prim_path.startswith("/World/") and not prim_path.startswith("/World/ground/terrain/"):
            candidate_paths.append("/World/ground/terrain/" + prim_path[len("/World/"):])
        self.prim = next(
            (candidate for path in candidate_paths if (candidate := stage.GetPrimAtPath(path)).IsValid()),
            stage.GetPrimAtPath(prim_path),
        )
        if not self.prim or not self.prim.IsValid():
            raise RuntimeError(f"phase-one vehicle prim not found; tried {candidate_paths}")
        self.prim_path = str(self.prim.GetPath())
        self.prim.Load()
        if not self.prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"phase-one vehicle lacks RigidBodyAPI: {self.prim_path}")
        if self.prim.GetAttribute("physics:kinematicEnabled").Get() is not True:
            raise RuntimeError(f"phase-one vehicle is not kinematic: {self.prim_path}")
        xformable = UsdGeom.Xformable(self.prim)
        translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        orient_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
        if len(translate_ops) != 1 or len(orient_ops) != 1:
            raise RuntimeError(
                f"phase-one vehicle requires one translate and orient op; got {len(translate_ops)}, {len(orient_ops)}"
            )
        self.translate_op = translate_ops[0]
        self.orient_op = orient_ops[0]
        self.initial_translate = Gf.Vec3d(self.translate_op.Get())
        self.initial_orient = self.orient_op.Get()
        rigid_body = UsdPhysics.RigidBodyAPI(self.prim)
        self.velocity_attr = rigid_body.GetVelocityAttr()
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        ).ComputeWorldBound(self.prim).ComputeAlignedRange()
        self.original_center = np.asarray(bbox.GetMidpoint(), dtype=np.float64)
        self.size = np.asarray(bbox.GetSize(), dtype=np.float64)
        self.half_extents_xy = np.maximum(self.size[:2] / 2.0, np.asarray([0.5, 0.3]))
        import omni.physics.tensors.impl.api as physx
        import torch

        self.physics_sim_view = physx.create_simulation_view("torch")
        self.physics_sim_view.set_subspace_roots("/")
        self.physics_sim_view.initialize_kinematic_bodies()
        self.rigid_body_view = self.physics_sim_view.create_rigid_body_view(self.prim_path)
        if self.rigid_body_view.count != 1:
            raise RuntimeError(
                f"phase-one vehicle PhysX view expected one body, got {self.rigid_body_view.count}: {self.prim_path}"
            )
        self.initial_physx_transform = self.rigid_body_view.get_transforms().clone()
        self.rigid_body_indices = torch.arange(
            self.rigid_body_view.count,
            dtype=torch.long,
            device=self.initial_physx_transform.device,
        )
        self.visual_clone_path = "/UrbanVerseEvaluation/Phase1OriginalVehicleVisual"
        self.visual_clone = stage.DefinePrim(self.visual_clone_path, "Xform")
        source_payload = None
        source_payload_layer = None
        for prim_spec in self.prim.GetPrimStack():
            items = list(prim_spec.payloadList.GetAddedOrExplicitItems())
            if items:
                source_payload = items[0]
                source_payload_layer = prim_spec.layer
                break
        if source_payload is None or source_payload_layer is None:
            raise RuntimeError("phase-one vehicle payload could not be resolved for the visual representation")
        resolved_payload_path = Sdf.ComputeAssetPathRelativeToLayer(
            source_payload_layer, source_payload.assetPath
        )
        self.visual_clone.GetPayloads().AddPayload(
            resolved_payload_path,
            source_payload.primPath,
            source_payload.layerOffset,
        )
        self.visual_clone.Load()
        UsdGeom.Imageable(self.prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdGeom.Imageable(self.visual_clone).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        visual_descendants = list(Usd.PrimRange(self.visual_clone))
        for item in visual_descendants:
            if item.HasAPI(UsdPhysics.RigidBodyAPI):
                item.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if item.HasAPI(UsdPhysics.CollisionAPI):
                item.RemoveAPI(UsdPhysics.CollisionAPI)
        visual_xformable = UsdGeom.Xformable(self.visual_clone)
        visual_xformable.ClearXformOpOrder()
        self.visual_translate_op = visual_xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        orient_precision = (
            UsdGeom.XformOp.PrecisionDouble
            if type(self.initial_orient).__name__ == "Quatd"
            else UsdGeom.XformOp.PrecisionFloat
        )
        self.visual_orient_op = visual_xformable.AddOrientOp(orient_precision)
        visual_local_range = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        ).ComputeLocalBound(self.visual_clone).ComputeAlignedRange()
        self.visual_local_center = np.asarray(visual_local_range.GetMidpoint(), dtype=np.float64)
        base_quat_xyzw = self.initial_physx_transform[0, 3:7].detach().cpu().numpy().astype(np.float64)
        z_alignment_xyzw = np.asarray([0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)], dtype=np.float64)
        visual_quat = self._multiply_quaternions_xyzw(z_alignment_xyzw, base_quat_xyzw)
        # GLB vehicle assets do not all share the same internal up-axis.  The
        # d90c Jeep is already upright after the common Z-up alignment, while
        # the older 403d SUV required an additional 180-degree X correction.
        self.extra_upright_flip_applied = "d90c7f830f9c41398bb55de4a2e001be" not in self.prim_path
        if self.extra_upright_flip_applied:
            upright_flip_xyzw = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            visual_quat = self._multiply_quaternions_xyzw(upright_flip_xyzw, visual_quat)
        self.visual_quat_xyzw = visual_quat
        # The d90c payload's local bounds are offset from the authored scene
        # vehicle/collision bounds.  This measured root-space correction puts
        # the rendered body center on the physical vehicle center.  Keep it
        # scoped to this qualified asset; other vehicles need their own probe.
        self.visual_root_position_correction = np.zeros(3, dtype=np.float64)
        if "d90c7f830f9c41398bb55de4a2e001be" in self.prim_path:
            self.visual_root_position_correction = np.asarray(
                [0.0, 2.52999997523385, 0.4900001287460329], dtype=np.float64
            )
        descendants = list(Usd.PrimRange(self.prim))
        self.evidence = {
            "requested_prim_path": requested_prim_path,
            "resolved_prim_path": self.prim_path,
            "active": self.prim.IsActive(),
            "loaded": self.prim.IsLoaded(),
            "descendant_count": len(descendants),
            "mesh_count": sum(item.IsA(UsdGeom.Mesh) for item in descendants),
            "collision_api_count": sum(item.HasAPI(UsdPhysics.CollisionAPI) for item in descendants),
            "has_rigid_body_api": True,
            "kinematic_enabled": True,
            "original_world_bbox_center": self.original_center.tolist(),
            "world_bbox_size": self.size.tolist(),
            "session_layer": self.layer.identifier,
            "source_modified": False,
            "visual_representation": {
                "path": self.visual_clone_path,
                "source_payload": resolved_payload_path,
                "rigid_body_api_removed": True,
                "collision_api_removed": True,
                "extra_upright_flip_applied": self.extra_upright_flip_applied,
                "root_position_correction_xyz": self.visual_root_position_correction.tolist(),
            },
        }
        self.last_state: dict[str, Any] | None = None

    def _quaternion(self, heading_rad: float):
        from pxr import Gf

        real = math.cos(heading_rad / 2.0)
        imaginary = Gf.Vec3f(0.0, 0.0, math.sin(heading_rad / 2.0))
        if type(self.initial_orient).__name__ == "Quatd":
            return Gf.Quatd(real, Gf.Vec3d(imaginary))
        return Gf.Quatf(real, imaginary)

    def update(self, timestamp_s: float) -> dict[str, Any]:
        from pxr import Gf

        state = self.schedule.sample(timestamp_s)
        desired_center = np.asarray(
            [state["center_xy"][0], state["center_xy"][1], self.original_center[2]], dtype=np.float64
        )
        offset = desired_center - self.original_center
        translate = np.asarray(self.initial_translate, dtype=np.float64) + offset
        self.translate_op.Set(Gf.Vec3d(*translate.tolist()))
        # This source vehicle's mesh vertices retain large scene-space offsets.
        # Rotating the root around its distant authored origin swings the visual
        # and collider far away from the requested crossing.  Preserve the
        # original orientation and translate the intact vehicle whole-body,
        # matching the qualified scene-10 root-takeover probe.
        self.orient_op.Set(self.initial_orient)
        physics_transform = self.initial_physx_transform.clone()
        physics_transform[0, :3] += physics_transform.new_tensor(offset.tolist())
        visual_world_pose = self._visual_pose_for_center(desired_center)
        self._set_visual_world_pose(visual_world_pose)
        self.rigid_body_view.set_transforms(physics_transform, indices=self.rigid_body_indices)
        self.velocity_attr.Set(Gf.Vec3f(float(state["velocity_xy"][0]), float(state["velocity_xy"][1]), 0.0))
        self.last_state = {
            **state,
            "center_xy": np.asarray(state["center_xy"], dtype=np.float64),
            "velocity_xy": np.asarray(state["velocity_xy"], dtype=np.float64),
            "authored_translate": translate,
            "visual_world_pose_xyzw": visual_world_pose,
        }
        return self.last_state

    @staticmethod
    def _multiply_quaternions_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        result = np.asarray(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )
        return result / np.linalg.norm(result)

    @staticmethod
    def _rotate_vector_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
        xyz = quat[:3]
        scalar = quat[3]
        return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + scalar * vector)

    def _visual_pose_for_center(self, desired_center: np.ndarray) -> np.ndarray:
        rotated_local_center = self._rotate_vector_xyzw(self.visual_quat_xyzw, self.visual_local_center)
        root_position = (
            desired_center
            - rotated_local_center
            + self.visual_root_position_correction
        )
        return np.concatenate((root_position, self.visual_quat_xyzw))

    def _set_visual_world_pose(self, pose_xyzw: np.ndarray) -> None:
        from pxr import Gf

        self.visual_translate_op.Set(Gf.Vec3d(*pose_xyzw[:3].tolist()))
        real = float(pose_xyzw[6])
        imaginary = pose_xyzw[3:6]
        if type(self.initial_orient).__name__ == "Quatd":
            quat = Gf.Quatd(real, Gf.Vec3d(*imaginary.tolist()))
        else:
            quat = Gf.Quatf(real, Gf.Vec3f(*imaginary.tolist()))
        self.visual_orient_op.Set(quat)

    def refresh_visual_transform(self) -> None:
        """Re-author the visual root after PhysX write-back and before rendering."""
        if self.last_state is None:
            return
        from pxr import Gf

        translate = np.asarray(self.last_state["authored_translate"], dtype=np.float64)
        self.translate_op.Set(Gf.Vec3d(*translate.tolist()))
        self.orient_op.Set(self.initial_orient)
        self._set_visual_world_pose(np.asarray(self.last_state["visual_world_pose_xyzw"], dtype=np.float64))

    def serializable_state(self) -> dict[str, Any]:
        if self.last_state is None:
            raise RuntimeError("vehicle actor has not been sampled")
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in self.last_state.items()
        }

    def measure_world_geometry(self) -> dict[str, Any]:
        """Measure the composed geometry after authored transforms/PhysX updates."""
        from pxr import Gf, Usd, UsdGeom

        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        ).ComputeWorldBound(self.prim).ComputeAlignedRange()
        world_matrix = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        root_origin = world_matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        measurement = {
            "bbox_center": np.asarray(bbox.GetMidpoint(), dtype=np.float64).tolist(),
            "bbox_size": np.asarray(bbox.GetSize(), dtype=np.float64).tolist(),
            "root_world_origin": np.asarray(root_origin, dtype=np.float64).tolist(),
        }
        visual_descendants = list(Usd.PrimRange(self.visual_clone))
        visual_bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        ).ComputeWorldBound(self.visual_clone).ComputeAlignedRange()
        measurement["visual"] = {
            "descendant_count": len(visual_descendants),
            "mesh_count": sum(item.IsA(UsdGeom.Mesh) for item in visual_descendants),
            "bbox_center": np.asarray(visual_bbox.GetMidpoint(), dtype=np.float64).tolist(),
            "bbox_size": np.asarray(visual_bbox.GetSize(), dtype=np.float64).tolist(),
        }
        self.evidence["measured_world_geometry"] = measurement
        return measurement

    def summary(self) -> dict[str, Any]:
        return {
            **self.evidence,
            "schedule": {
                "conflict_xy": self.schedule.conflict_xy.tolist(),
                "direction_xy": self.schedule.direction_xy.tolist(),
                "heading_rad": self.schedule.heading_rad,
                "distance_m": self.schedule.distance,
                "speed_mps": self.schedule.speed,
                "start_time_s": self.schedule.start_time_s,
                "crossing_time_s": self.schedule.crossing_time_s,
                "end_time_s": self.schedule.end_time_s,
                "robot_observation_used": False,
            },
            "last_state": self.serializable_state() if self.last_state is not None else None,
        }
