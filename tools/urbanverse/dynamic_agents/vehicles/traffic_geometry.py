#!/usr/bin/env python3
"""Controlled four-car traffic flow with Lane support and OBB validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .scene10_straight_traffic import (
    BASE_QUAT_XYZW, QUALIFIED_ASSET_ID, command_output, contact_sheet,
    multiply_quaternions_xyzw, now, package_version, rgb_array,
    rotate_vector_xyzw, sha256, wait_stage, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("usd", "tar", "vehicle-asset", "registry", "audit-inventory", "run-dir", "experience", "ext-folder"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=30.0)
    return parser.parse_args()


def path_pose(arc_m: float) -> tuple[np.ndarray, float]:
    # Visible S turn within the northbound southern Lane branch, bounded by
    # the police/fire vehicles behind and the intersection cluster ahead.
    y = 463.0 + arc_m
    x = -622.5 + 1.0 * math.sin((y - 463.0) * 2.0 * math.pi / 12.0)
    dxdy = 1.0 * 2.0 * math.pi / 12.0 * math.cos((y - 463.0) * 2.0 * math.pi / 12.0)
    return np.asarray([x, y], dtype=np.float64), math.atan2(1.0, dxdy)


def footprint(center: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    local = np.asarray([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])
    c,s=math.cos(yaw),math.sin(yaw); rotation=np.asarray([[c,-s],[s,c]])
    return local @ rotation.T + center


def sat_intersects(left: np.ndarray, right: np.ndarray, margin: float = 0.0) -> bool:
    for polygon in (left, right):
        edges=np.roll(polygon,-1,axis=0)-polygon
        for edge in edges:
            axis=np.asarray([-edge[1],edge[0]],dtype=float); axis/=max(np.linalg.norm(axis),1e-12)
            a,b=left@axis,right@axis
            if a.max()+margin < b.min() or b.max()+margin < a.min(): return False
    return True


def polygon_clearance(left: np.ndarray, right: np.ndarray) -> float:
    if sat_intersects(left,right): return -0.0
    def point_segment(p,a,b):
        ab=b-a; t=float(np.clip(np.dot(p-a,ab)/max(np.dot(ab,ab),1e-12),0,1)); return float(np.linalg.norm(p-(a+t*ab)))
    return min(point_segment(p,a,b) for p in left for a,b in zip(right,np.roll(right,-1,axis=0)))


def annotate(frame: np.ndarray, timestamp: float, speeds: list[float], stop: bool) -> np.ndarray:
    image=Image.fromarray(frame); draw=ImageDraw.Draw(image,"RGBA")
    draw.rounded_rectangle((16,16,600,105),radius=9,fill=(8,12,18,180))
    draw.text((30,27),"scene_10 · controlled 4-car S-turn · static vehicles retained",fill=(255,255,255,255))
    draw.text((30,53),f"t={timestamp:05.1f}s  leader signal={'STOP' if stop else 'GO'}",fill=(255,190,83,255) if stop else (130,236,164,255))
    draw.text((30,78),"speeds: "+" / ".join(f"V{i+1} {v:.2f}" for i,v in enumerate(speeds))+" m/s",fill=(132,220,255,255))
    return np.asarray(image)


class LaneFootprint:
    """Direct XY containment queries against authored semantic Lane triangles."""

    def __init__(self, stage):
        from pxr import Usd, UsdGeom
        triangles=[]; paths=[]
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            path=str(prim.GetPath())
            if not prim.IsA(UsdGeom.Mesh) or "/Lane/Lane/" not in path or not path.endswith("/mesh"):
                continue
            mesh=UsdGeom.Mesh(prim); points=mesh.GetPointsAttr().Get(); counts=mesh.GetFaceVertexCountsAttr().Get(); indices=mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices: continue
            matrix=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            world=np.asarray([matrix.Transform(point) for point in points],dtype=np.float64)
            cursor=0
            for count in counts:
                face=list(indices[cursor:cursor+count]); cursor+=count
                for offset in range(1,len(face)-1): triangles.append(world[[face[0],face[offset],face[offset+1]]]); paths.append(path)
        if not triangles: raise RuntimeError("no authored Lane mesh triangles found")
        self.triangles=np.asarray(triangles,dtype=np.float64); self.xy=self.triangles[:,:,:2]
        self.minimum=self.xy.min(axis=1); self.maximum=self.xy.max(axis=1); self.paths=paths
        self.spatial_cell_size_m = 5.0
        self.spatial_buckets = {}
        for triangle_index, (lower, upper) in enumerate(zip(self.minimum, self.maximum)):
            first = np.floor((lower - 0.05) / self.spatial_cell_size_m).astype(np.int64)
            last = np.floor((upper + 0.05) / self.spatial_cell_size_m).astype(np.int64)
            for cell_x in range(int(first[0]), int(last[0]) + 1):
                for cell_y in range(int(first[1]), int(last[1]) + 1):
                    self.spatial_buckets.setdefault((cell_x, cell_y), []).append(triangle_index)
        self.query_count = 0
        self.candidate_count = 0

    def contains(self, point: np.ndarray, tolerance: float = 0.03) -> bool:
        point=np.asarray(point,dtype=np.float64)
        cell=tuple(np.floor(point/self.spatial_cell_size_m).astype(np.int64).tolist())
        bucket=np.asarray(self.spatial_buckets.get(cell, ()),dtype=np.int64)
        self.query_count += 1
        self.candidate_count += int(len(bucket))
        if not len(bucket): return False
        candidates=bucket[
            np.all(point>=self.minimum[bucket]-tolerance,axis=1)
            & np.all(point<=self.maximum[bucket]+tolerance,axis=1)
        ]
        if not len(candidates): return False
        tri=self.xy[candidates]; a=tri[:,0]; b=tri[:,1]; c=tri[:,2]
        v0=b-a; v1=c-a; v2=point-a
        denominator=v0[:,0]*v1[:,1]-v1[:,0]*v0[:,1]; valid=np.abs(denominator)>1e-10
        u=np.zeros_like(denominator); v=np.zeros_like(denominator)
        u[valid]=(v2[valid,0]*v1[valid,1]-v1[valid,0]*v2[valid,1])/denominator[valid]
        v[valid]=(v0[valid,0]*v2[valid,1]-v2[valid,0]*v0[valid,1])/denominator[valid]
        return bool(np.any(valid&(u>=-tolerance)&(v>=-tolerance)&(u+v<=1.0+tolerance)))

    def height(self, point: np.ndarray, tolerance: float = 0.03) -> float:
        """Interpolate Z from an authored Lane triangle at the queried XY."""
        point=np.asarray(point,dtype=np.float64)
        candidates=np.nonzero(np.all(point>=self.minimum-tolerance,axis=1)&np.all(point<=self.maximum+tolerance,axis=1))[0]
        values=[]
        for index in candidates:
            tri=self.triangles[index]; a,b,c=tri[:,:2]; v0=b-a; v1=c-a; v2=point-a
            denominator=v0[0]*v1[1]-v1[0]*v0[1]
            if abs(denominator)<1e-10: continue
            u=(v2[0]*v1[1]-v1[0]*v2[1])/denominator; v=(v0[0]*v2[1]-v2[0]*v0[1])/denominator
            if u>=-tolerance and v>=-tolerance and u+v<=1+tolerance:
                values.append(float(tri[0,2]+u*(tri[1,2]-tri[0,2])+v*(tri[2,2]-tri[0,2])))
        if not values: raise RuntimeError(f"no Lane surface at {point.tolist()}")
        return float(np.median(values))


class Vehicle:
    def __init__(self, stage, asset: Path, index: int, initial_arc: float, length: float, width: float, center_height: float):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        self.index=index; self.arc=initial_arc; self.speed=0.0; self.length=length; self.width=width; self.center_height=center_height
        self.root=stage.DefinePrim(f"/ControlledTraffic/Vehicle_{index:02d}","Xform"); self.root.GetPayloads().AddPayload(str(asset)); self.root.Load()
        for prim in Usd.PrimRange(self.root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI): prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI): prim.RemoveAPI(UsdPhysics.CollisionAPI)
        xform=UsdGeom.Xformable(self.root); xform.ClearXformOpOrder()
        self.translate=xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble,"traffic")
        self.orient=xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,"traffic")
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render],useExtentsHint=True).ComputeLocalBound(self.root).ComputeAlignedRange()
        self.local_center=np.asarray(bounds.GetMidpoint(),dtype=float)

    def set_pose(self, center_xy: np.ndarray, yaw: float, ground_z: float):
        from pxr import Gf
        center=np.asarray([*center_xy,ground_z+self.center_height]); yaw_q=np.asarray([0.,0.,math.sin(yaw/2),math.cos(yaw/2)])
        quat=multiply_quaternions_xyzw(yaw_q,BASE_QUAT_XYZW); root_xyz=center-rotate_vector_xyzw(quat,self.local_center)
        self.translate.Set(Gf.Vec3d(*root_xyz.tolist())); self.orient.Set(Gf.Quatd(float(quat[3]),Gf.Vec3d(*quat[:3].tolist())))
        return center, footprint(center_xy,yaw,self.length,self.width), root_xyz


def main() -> int:
    args=parse_args(); source=args.usd.resolve(); source_tar=args.tar.resolve(); asset=args.vehicle_asset.resolve(); registry_path=args.registry.resolve(); audit_path=args.audit_inventory.resolve(); run_dir=args.run_dir.resolve()
    metadata,captures,visualizations=run_dir/"metadata",run_dir/"captures",run_dir/"visualizations"
    for directory in (metadata,captures,visualizations): directory.mkdir(parents=True,exist_ok=False)
    sources={"usd":sha256(source),"tar":sha256(source_tar),"asset":sha256(asset),"registry":sha256(registry_path),"audit":sha256(audit_path)}
    wrapper=run_dir/"scene10_controlled_multicar_wrapper.usda"; wrapper.write_text(f'#usda 1.0\n(\n subLayers=[@{source.as_posix()}@]\n)\n',encoding="utf-8")
    environment={"timestamp":now(),"git_commit":command_output(["git","rev-parse","HEAD"]),"git_status_short":command_output(["git","status","--short"]),"command_line":sys.argv,"physical_gpu_index":args.gpu,"nvidia_smi":command_output(["nvidia-smi"]),"isaac_sim":package_version("isaacsim"),"replicator":package_version("isaacsim-replicator"),"kit":package_version("isaacsim-kernel"),"renderer":"RayTracedLighting","resolution":[args.width,args.height],"camera":"/ControlledTraffic/Camera","source_usd":str(source),"source_tar":str(source_tar),"registry":str(registry_path),"source_asset_hashes":sources}
    write_json(metadata/"environment.json",environment); write_json(metadata/"summary.json",{"status":"running","started_at":now()})
    app=annotator=render_product=timeline=None; started=time.perf_counter()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({"headless":True,"renderer":"RayTracedLighting","width":args.width,"height":args.height,"active_gpu":args.gpu,"physics_gpu":args.gpu,"multi_gpu":False,"max_gpu_count":1,"create_new_stage":False,"extra_args":["--ext-folder",str(args.ext_folder.resolve()),"--/renderer/multiGpu/enabled=false","--/app/window/hideUi=1"]},experience=str(args.experience.resolve()))
        import carb,cv2,omni.kit.app,omni.physx,omni.replicator.core as rep,omni.timeline,omni.usd
        from pxr import Gf,UsdGeom
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.asset_converter",True)
        for _ in range(4): app.update()
        context=omni.usd.get_context()
        if not context.open_stage(str(wrapper)): raise RuntimeError("wrapper open failed")
        wait_stage(context,app); stage=context.get_stage(); stage.SetEditTarget(stage.GetRootLayer()); carb.settings.get_settings().set("/rtx/post/tonemap/exposure",0.0)
        registry=json.loads(registry_path.read_text()); record=next(x for x in registry["records"] if x["asset_id"]==QUALIFIED_ASSET_ID)
        length,width=map(float,record["footprint"]["length_width_m"]); center_height=float(record["grounding"]["body_center_height_above_support_m"])
        static=[]
        for item in json.loads(audit_path.read_text()):
            if item["quality"]["status"]=="reject" or not item.get("obb"): continue
            static.append((item["path"],np.asarray(item["obb"]["corners_xy"],dtype=float)))
        known_static_paths={path for path,_ in static}
        bbox_cache=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render],useExtentsHint=True)
        # Include non-private traffic assets (police/emergency/service cars)
        # omitted by the original private-vehicle-only inventory.
        vehicle_tokens=("vehicle", "police_car", "bus", "truck", "suv", "pickup", "van")
        for prim in stage.Traverse():
            path=str(prim.GetPath())
            name=prim.GetName().lower()
            if path in known_static_paths or path.count("/") != 2 or not any(token in name for token in vehicle_tokens):
                continue
            bounds=bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo=np.asarray(bounds.GetMin(),dtype=float); hi=np.asarray(bounds.GetMax(),dtype=float); size=hi-lo
            if not np.all(np.isfinite(size)) or np.any(size<=0.05) or np.max(size[:2])>30.0:
                continue
            static.append((path,np.asarray([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],dtype=float)))
        lane=LaneFootprint(stage)
        # Reject an invalid route before paying the cost of a 300-frame RTX
        # render.  Dense arc sampling covers every vehicle because all four
        # share the same centerline and footprint.
        preflight_min_static=math.inf
        for arc in np.linspace(0.0,22.0,221):
            center_xy,yaw=path_pose(float(arc)); poly=footprint(center_xy,yaw,length,width)
            if not all(lane.contains(corner) for corner in poly):
                raise RuntimeError(f"route preflight leaves Lane at arc={arc:.2f}, center={center_xy.tolist()}")
            for path,static_poly in static:
                clearance=polygon_clearance(poly,static_poly); preflight_min_static=min(preflight_min_static,clearance)
                if sat_intersects(poly,static_poly):
                    raise RuntimeError(f"route preflight intersects static vehicle {path} at arc={arc:.2f}")
        vehicles=[Vehicle(stage,asset,i,initial,length,width,center_height) for i,initial in enumerate((18.0,12.0,6.0,0.0),start=1)]
        timeline=omni.timeline.get_timeline_interface(); timeline.play()
        for _ in range(24): app.update()
        query=omni.physx.get_physx_scene_query_interface()
        target=Gf.Vec3d(-622.5,474.0,1.0); eye=target+Gf.Vec3d(24.,-28.,27.)
        camera=UsdGeom.Camera.Define(stage,"/ControlledTraffic/Camera"); camera.CreateFocalLengthAttr(27.0); camera.CreateHorizontalApertureAttr(24.0); camera.CreateClippingRangeAttr(Gf.Vec2f(.1,100000.))
        cam=UsdGeom.Xformable(camera.GetPrim()); cam.ClearXformOpOrder(); cam.AddTransformOp().Set(Gf.Matrix4d(1).SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse())
        render_product=rep.create.render_product(camera.GetPath(),(args.width,args.height),force_new=True); annotator=rep.AnnotatorRegistry.get_annotator("rgb"); annotator.attach([render_product])
        for _ in range(30): app.update()
        rep.orchestrator.step(rt_subframes=2)
        count=int(round(args.duration*args.fps)); dt=1/args.fps; video=captures/"scene10_controlled_multicar_traffic.webm"; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"VP90"),args.fps,(args.width,args.height))
        if not writer.isOpened(): raise RuntimeError("video writer failed")
        states=[]; lane_failures=[]; dynamic_events=[]; static_events=[]; min_dynamic=math.inf; min_static=math.inf; min_gap=math.inf; max_gap=-math.inf; key_paths=[]; key_indices=sorted(set((0,count//4,count//2,3*count//4,count-1)))
        try:
            for frame_index in range(count):
                timestamp=frame_index*dt; signal_stop=10.0<=timestamp<15.0
                for i,vehicle in enumerate(vehicles):
                    desired=1.65
                    if i==0 and ((signal_stop and vehicle.arc>=20.0) or vehicle.arc>=22.0): desired=0.0
                    if i>0:
                        gap=vehicles[i-1].arc-vehicle.arc-length
                        desired=min(desired,max(0.0,0.42*(gap-2.3))+vehicles[i-1].speed)
                    accel=1.0 if desired>vehicle.speed else 2.0; vehicle.speed+=float(np.clip(desired-vehicle.speed,-accel*dt,accel*dt)); vehicle.arc+=vehicle.speed*dt
                frame_states=[]; polygons=[]
                for vehicle in vehicles:
                    center_xy,yaw=path_pose(vehicle.arc); ray=query.raycast_closest(carb.Float3(float(center_xy[0]),float(center_xy[1]),3.),carb.Float3(0,0,-1),8.)
                    if not ray.get("hit") or ray.get("position") is None: raise RuntimeError(f"center ground miss {center_xy}")
                    ground_z=float(ray["position"][2]); center,poly,root_xyz=vehicle.set_pose(center_xy,yaw,ground_z); polygons.append(poly)
                    support=[]
                    for corner in poly:
                        valid=lane.contains(corner)
                        support.append({"xy":corner.tolist(),"valid_lane_support":valid,"method":"authored_lane_triangle_xy"})
                        if not valid and len(lane_failures)<100: lane_failures.append({"frame":frame_index,"vehicle":vehicle.index,"xy":corner.tolist(),"method":"authored_lane_triangle_xy"})
                    rep.orchestrator.step(rt_subframes=1)
                    bound=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render],useExtentsHint=True).ComputeWorldBound(vehicle.root).ComputeAlignedRange(); gap=float(bound.GetMin()[2])-ground_z; min_gap=min(min_gap,gap); max_gap=max(max_gap,gap)
                    frame_states.append({"vehicle_id":vehicle.index,"arc_m":vehicle.arc,"speed_mps":vehicle.speed,"center_xyz":center.tolist(),"heading_deg":math.degrees(yaw),"obb_corners_xy":poly.tolist(),"lane_support":support,"visible_bottom_ground_gap_m":gap,"root_xyz":root_xyz.tolist()})
                for i in range(len(polygons)):
                    for j in range(i+1,len(polygons)):
                        clearance=polygon_clearance(polygons[i],polygons[j]); min_dynamic=min(min_dynamic,clearance)
                        if sat_intersects(polygons[i],polygons[j]) and len(dynamic_events)<100: dynamic_events.append({"frame":frame_index,"vehicles":[i+1,j+1]})
                    for path,static_poly in static:
                        clearance=polygon_clearance(polygons[i],static_poly); min_static=min(min_static,clearance)
                        if sat_intersects(polygons[i],static_poly) and len(static_events)<100: static_events.append({"frame":frame_index,"vehicle":i+1,"static":path})
                states.append({"frame":frame_index,"timestamp_s":timestamp,"signal_stop":signal_stop,"vehicles":frame_states})
                raw=rgb_array(annotator); frame=annotate(raw,timestamp,[v.speed for v in vehicles],signal_stop); writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                if frame_index in key_indices:
                    path=captures/f"frame_{frame_index:04d}.png"; Image.fromarray(frame).save(path); key_paths.append(path)
        finally: writer.release()
        write_json(metadata/"traffic_states.json",states); contact_sheet(key_paths,visualizations/"controlled_multicar_contact_sheet.png")
        capture=cv2.VideoCapture(str(video)); validation={"opened":bool(capture.isOpened()),"frame_count":int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),"width":int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),"height":int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),"fps":capture.get(cv2.CAP_PROP_FPS),"file_size_bytes":video.stat().st_size}; capture.release()
        after={"usd":sha256(source),"tar":sha256(source_tar),"asset":sha256(asset),"registry":sha256(registry_path),"audit":sha256(audit_path)}
        if after!=sources: raise RuntimeError("source hash changed")
        passed=not lane_failures and not dynamic_events and not static_events and min_gap>-0.03 and max_gap<0.03 and validation["frame_count"]==count
        summary={"status":"passed" if passed else "failed_validation","completed_at":now(),"elapsed_s":time.perf_counter()-started,"dynamic_vehicle_count":4,"static_vehicle_count":len(static),"duration_s":args.duration,"fps":args.fps,"frame_count":count,"lane_triangle_count":int(len(lane.triangles)),"lane_support_method":"direct authored Lane triangle XY containment","route_preflight_minimum_static_obb_clearance_m":preflight_min_static,"lane_support_failure_count":len(lane_failures),"dynamic_collision_event_count":len(dynamic_events),"static_collision_event_count":len(static_events),"minimum_dynamic_obb_clearance_m":min_dynamic,"minimum_static_obb_clearance_m":min_static,"minimum_ground_gap_m":min_gap,"maximum_ground_gap_m":max_gap,"video_validation":validation,"source_modified":False,"claim_scope":"controlled kinematic multicar flow with direct Lane-mesh support and OBB validation; no wheel dynamics"}
        write_json(metadata/"validation_events.json",{"lane":lane_failures,"dynamic":dynamic_events,"static":static_events}); write_json(metadata/"summary.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if passed else 2
    except Exception as exc:
        write_json(metadata/"summary.json",{"status":"failed","completed_at":now(),"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}); print(traceback.format_exc(),file=sys.stderr); return 1
    finally:
        if timeline is not None: timeline.stop()
        if annotator is not None: annotator.detach()
        if app is not None: app.close()


if __name__=="__main__": raise SystemExit(main())
