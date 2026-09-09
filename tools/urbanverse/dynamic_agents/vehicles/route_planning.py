#!/usr/bin/env python3
"""Long multi-route native-vehicle demo with CPU disc ORCA avoidance."""

from __future__ import annotations

import argparse, heapq, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from .traffic_geometry import (
    LaneFootprint, Vehicle, footprint, polygon_clearance, sat_intersects,
)
from .scene10_straight_traffic import (
    QUALIFIED_ASSET_ID, command_output, contact_sheet, now, package_version,
    rgb_array, sha256, wait_stage, write_json,
)
from ..core.orca import step_orca
from ..core.vehicle_pose import recentered_root_matrix_2d

STATIC_SAFETY_MARGIN_M = 0.25
RASTER_UNCERTAINTY_MARGIN_M = 0.40
MIXED_VEHICLE_IDS = (
    "955edc733c6d44fabc0ad7c246a15896",  # car
    "0e249864d1df49818ef2c03c336a934c",  # convertible
    "1f614e803fc24946b83bdc2dd9926d62",  # coupe
    "c640e4e7c68545e09a9348494c2c13a1",  # pickup
    "09ebc68dcb634807a9dace7404ce5e66",  # SUV
)


def physical_road_height(scene_query, center_xy):
    """Return the visible/collision road surface, not the offset semantic Lane."""
    import carb
    ray = scene_query.raycast_closest(
        carb.Float3(float(center_xy[0]), float(center_xy[1]), 8.0),
        carb.Float3(0.0, 0.0, -1.0), 20.0,
    )
    if not isinstance(ray, dict) or not bool(ray.get("hit", False)) or ray.get("position") is None:
        raise RuntimeError(f"physical road ray missed at {np.asarray(center_xy).tolist()}")
    hit_text = f"{ray.get('collision', '')} {ray.get('rigidBody', '')}".lower()
    vehicle_tokens = ("vehicle", "police_car", "bus", "truck", "suv", "pickup", "van")
    if any(token in hit_text for token in vehicle_tokens):
        raise RuntimeError(f"physical road ray hit a vehicle at {np.asarray(center_xy).tolist()}: {hit_text}")
    return float(ray["position"][2]), str(ray.get("collision") or "")


def args_parse():
    p=argparse.ArgumentParser()
    for name in ("usd","tar","vehicle-asset","registry","audit-inventory","run-dir","experience","ext-folder"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--gpu",type=int,required=True); p.add_argument("--width",type=int,default=1600); p.add_argument("--height",type=int,default=900)
    p.add_argument("--fps",type=float,default=12.0); p.add_argument("--duration",type=float,default=140.0); p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--arrival-radius",type=float,default=.55); p.add_argument("--stall-timeout",type=float,default=40.0)
    p.add_argument("--frames",type=int)
    return p.parse_args()


def point_poly_clearance(point, poly):
    def segment(p,a,b):
        ab=b-a; t=float(np.clip(np.dot(p-a,ab)/max(np.dot(ab,ab),1e-12),0,1)); return float(np.linalg.norm(p-(a+t*ab)))
    return min(segment(point,a,b) for a,b in zip(poly,np.roll(poly,-1,axis=0)))


def largest_four_component(valid):
    valid=np.asarray(valid,bool); seen=np.zeros_like(valid); best=[]; ny,nx=valid.shape
    for j in range(ny):
        for i in range(nx):
            if not valid[j,i] or seen[j,i]: continue
            stack=[(j,i)]; seen[j,i]=True; cells=[]
            while stack:
                y,x=stack.pop(); cells.append((y,x))
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    yy,xx=y+dy,x+dx
                    if 0<=yy<ny and 0<=xx<nx and valid[yy,xx] and not seen[yy,xx]: seen[yy,xx]=True; stack.append((yy,xx))
            if len(cells)>len(best): best=cells
    out=np.zeros_like(valid)
    for j,i in best: out[j,i]=True
    return out


def static_vehicles(stage,audit_path):
    from pxr import Usd,UsdGeom
    static=[]
    for item in json.loads(audit_path.read_text()):
        if item["quality"]["status"]!="reject" and item.get("obb"): static.append((item["path"],np.asarray(item["obb"]["corners_xy"],float)))
    known={p for p,_ in static}; cache=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render],useExtentsHint=True)
    tokens=("vehicle","police_car","bus","truck","suv","pickup","van")
    for prim in stage.Traverse():
        path=str(prim.GetPath()); name=prim.GetName().lower()
        source_equivalent = path.replace("/World/ground/terrain", "/World", 1)
        if path in known or source_equivalent in known or not prim.IsA(UsdGeom.Xform) or not any(t in name for t in tokens): continue
        # Some public-vehicle roots contain multiple spatially separated buses.
        # Represent every visible mesh body independently instead of wrapping
        # the whole root in one misleading box.
        bodies=[]
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Mesh): continue
            bounds=cache.ComputeWorldBound(child).ComputeAlignedRange(); lo=np.asarray(bounds.GetMin(),float); hi=np.asarray(bounds.GetMax(),float); size=hi-lo
            planar=np.sort(size[:2])
            vehicle_scale=(0.8<planar[0]<8.0 and 2.0<planar[1]<16.0 and 0.45<size[2]<5.0 and lo[2]<2.0)
            if np.all(np.isfinite(size)) and vehicle_scale:
                bodies.append((str(child.GetPath()),lo,hi))
        for body_path,lo,hi in bodies:
            static.append((body_path,np.asarray([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],float)))
    return static


class NativeVehicle:
    """Move an authored scene vehicle around its calibrated visible center."""
    def __init__(self,stage,record):
        from pxr import Gf,Usd,UsdGeom,UsdPhysics
        self.record=record; self.prim=stage.GetPrimAtPath(record["scene_prim_path"])
        if not self.prim.IsValid(): raise RuntimeError(f"missing native vehicle {record['scene_prim_path']}")
        for child in Usd.PrimRange(self.prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI): child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if child.HasAPI(UsdPhysics.CollisionAPI): child.RemoveAPI(UsdPhysics.CollisionAPI)
        xf=UsdGeom.Xformable(self.prim); self.original=np.asarray(xf.ComputeLocalToWorldTransform(0),float).T
        original_bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        ).ComputeWorldBound(self.prim).ComputeAlignedRange()
        parent_xf=UsdGeom.Xformable(self.prim.GetParent())
        self.parent_world=np.asarray(parent_xf.ComputeLocalToWorldTransform(0),float).T if parent_xf else np.eye(4)
        xf.ClearXformOpOrder(); self.op=xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble,"mixedTraffic")
        self.registry_original_center=np.asarray(
            record["body_frame"]["original_visible_center_xyz"], dtype=float
        )
        # A direct-payload visual proxy does not inherit the source prim's
        # calibrated root-local centre.  Measure the composed proxy itself;
        # otherwise the GLB's large internal coordinate offset can displace the
        # rendered body by more than 100 m from the planner centre.
        self.original_center=np.asarray(original_bounds.GetMidpoint(),dtype=float)
        original_center_h=np.r_[self.original_center,1.0]
        self.local_body_center=(np.linalg.inv(self.original)@original_center_h)[:3]
        self.initial_heading=float(
            record["front_direction"].get(
                "runtime_proxy_heading_deg", record["front_direction"]["heading_deg"]
            )
        )

    def set_pose(self,center_xy,yaw,ground_z):
        target=np.asarray([center_xy[0],center_xy[1],ground_z+float(self.record["grounding"]["body_center_height_above_support_m"])])
        world_matrix=recentered_root_matrix_2d(self.original,self.original_center[:2],center_xy,yaw-math.radians(self.initial_heading))
        world_matrix[2,3]+=target[2]-self.original_center[2]
        self.last_world_matrix = world_matrix.copy()
        self._author_world_matrix(world_matrix)
        length,width=map(float,self.record["footprint"]["length_width_m"])
        return target,footprint(center_xy,yaw,length,width),world_matrix[:3,3]

    def _author_world_matrix(self, world_matrix):
        from pxr import Gf

        # Xform ops are local to the prim's parent. Convert the requested world
        # transform back to that local space before writing it.
        matrix=np.linalg.inv(self.parent_world)@world_matrix
        # The adapter uses column-vector matrices; Gf/USD stores affine
        # transforms in row-vector convention (translation in the last row).
        # Transpose exactly once when authoring the xform op.
        usd_matrix=matrix.T
        authored=Gf.Matrix4d(1.0)
        for row in range(4):
            authored.SetRow(row,Gf.Vec4d(*usd_matrix[row].tolist()))
        self.op.Set(authored)

    def refresh_visual_transform(self):
        """Re-author the last pose after the physics step and before camera capture."""
        if hasattr(self, "last_world_matrix"):
            self._author_world_matrix(self.last_world_matrix)


def build_clearance_fields(lane,static,cell=.75):
    """Rasterize the road and obstacles once, then derive metric clearance."""
    from scipy.ndimage import distance_transform_edt
    xs=np.arange(-708.,-548.+1e-6,cell); ys=np.arange(448.,535.+1e-6,cell)
    road=np.asarray([[lane.contains(np.asarray([x,y])) for x in xs] for y in ys],bool)
    obstacle_image=Image.new("1",(len(xs),len(ys)),0); draw=ImageDraw.Draw(obstacle_image)
    for _,poly in static:
        pixels=[((float(p[0])-xs[0])/cell,(float(p[1])-ys[0])/cell) for p in poly]
        draw.polygon(pixels,fill=1)
    obstacles=np.asarray(obstacle_image,dtype=bool)
    # EDT measures between pixel centres. Subtract half a cell so the reported
    # clearance is conservative with respect to an unknown sub-cell boundary.
    road_clearance=np.maximum(0.,distance_transform_edt(road)*cell-cell/2)
    obstacle_clearance=np.maximum(0.,distance_transform_edt(~obstacles)*cell-cell/2)
    return xs,ys,road_clearance,obstacle_clearance


def build_grid(road_clearance,obstacle_clearance,half_width):
    """Build a per-vehicle center mask using lateral rather than diagonal size."""
    lateral_radius=half_width+STATIC_SAFETY_MARGIN_M
    return largest_four_component((road_clearance>=lateral_radius)&(obstacle_clearance>=lateral_radius+RASTER_UNCERTAINTY_MARGIN_M))


def point_segment_clearance(point,a,b):
    ab=b-a; t=float(np.clip(np.dot(point-a,ab)/max(np.dot(ab,ab),1e-12),0,1)); return float(np.linalg.norm(point-(a+t*ab)))


def segments_intersect(a,b,c,d):
    def cross(u,v): return float(u[0]*v[1]-u[1]*v[0])
    def on_segment(p,q,r):
        return min(p[0],r[0])-1e-10<=q[0]<=max(p[0],r[0])+1e-10 and min(p[1],r[1])-1e-10<=q[1]<=max(p[1],r[1])+1e-10
    o1=cross(b-a,c-a); o2=cross(b-a,d-a); o3=cross(d-c,a-c); o4=cross(d-c,b-c)
    if o1*o2 < -1e-10 and o3*o4 < -1e-10: return True
    if abs(o1)<=1e-10 and on_segment(a,c,b): return True
    if abs(o2)<=1e-10 and on_segment(a,d,b): return True
    if abs(o3)<=1e-10 and on_segment(c,a,d): return True
    if abs(o4)<=1e-10 and on_segment(c,b,d): return True
    return False


def segment_polygon_clearance(a,b,poly):
    for c,d in zip(poly,np.roll(poly,-1,axis=0)):
        if segments_intersect(a,b,c,d): return 0.0
    return min([point_segment_clearance(p,a,b) for p in poly]+[point_segment_clearance(a,c,d) for c,d in zip(poly,np.roll(poly,-1,axis=0))]+[point_segment_clearance(b,c,d) for c,d in zip(poly,np.roll(poly,-1,axis=0))])


def capsule_transition_clear(mask,xs,ys,a,b,length):
    """Validate the swept vehicle body along one directed grid edge.

    The longitudinal axis is extended by half the vehicle length and swept with
    a radius equal to half the width plus safety. This is substantially tighter
    than a half-diagonal disc while remaining conservative for straight edges.
    """
    a=np.asarray(a,float); b=np.asarray(b,float); delta=b-a; distance=float(np.linalg.norm(delta))
    if distance<1e-9: return False
    direction=delta/distance; axis_a=a-direction*(length/2); axis_b=b+direction*(length/2)
    count=max(2,int(math.ceil(float(np.linalg.norm(axis_b-axis_a))/(min(xs[1]-xs[0],ys[1]-ys[0])/3)))+1)
    for t in np.linspace(0.,1.,count):
        point=axis_a*(1-t)+axis_b*t
        i=int(round((point[0]-xs[0])/(xs[1]-xs[0]))); j=int(round((point[1]-ys[0])/(ys[1]-ys[0])))
        if not (0<=j<mask.shape[0] and 0<=i<mask.shape[1] and mask[j,i]): return False
    return True


def exact_capsule_transition_clear(static,a,b,length,width,safety=STATIC_SAFETY_MARGIN_M):
    """Exact static-OBB validation for a directed straight vehicle sweep."""
    a=np.asarray(a,float); b=np.asarray(b,float); delta=b-a; distance=float(np.linalg.norm(delta))
    if distance<1e-9: return False
    direction=delta/distance; axis_a=a-direction*(length/2); axis_b=b+direction*(length/2)
    radius=width/2+safety
    return not any(segment_polygon_clearance(axis_a,axis_b,poly)<radius for _,poly in static)


def astar_capsule(mask,start,goal,xs,ys,length,width=None,static=None):
    """Eight-neighbour A* whose edges admit a swept vehicle capsule."""
    if not mask[start] or not mask[goal]: return []
    frontier=[(0.,0.,start)]; parent={}; cost={start:0.}; transition_cache={}; ny,nx=mask.shape
    while frontier:
        _,current_cost,current=heapq.heappop(frontier)
        if current==goal:
            path=[current]
            while path[-1]!=start: path.append(parent[path[-1]])
            return list(reversed(path))
        if current_cost>cost.get(current,math.inf)+1e-9: continue
        cj,ci=current
        for dj in (-1,0,1):
            for di in (-1,0,1):
                if not (di or dj): continue
                nj,ni=cj+dj,ci+di
                if not (0<=nj<ny and 0<=ni<nx and mask[nj,ni]): continue
                if di and dj and (not mask[cj,ni] or not mask[nj,ci]): continue
                node=(nj,ni); key=tuple(sorted((current,node)))
                if key not in transition_cache:
                    a=np.asarray([xs[ci],ys[cj]]); b=np.asarray([xs[ni],ys[nj]])
                    transition_cache[key]=capsule_transition_clear(mask,xs,ys,a,b,length) and (static is None or exact_capsule_transition_clear(static,a,b,length,width))
                if not transition_cache[key]: continue
                step=math.sqrt(2.) if di and dj else 1.; candidate=current_cost+step
                if candidate+1e-9>=cost.get(node,math.inf): continue
                cost[node]=candidate; parent[node]=current
                heapq.heappush(frontier,(candidate+math.hypot(goal[0]-nj,goal[1]-ni),candidate,node))
    return []


def capsule_reachable_mask(mask,xs,ys,length):
    """Keep the largest component connected by valid swept-capsule edges."""
    neighbours={}; ny,nx=mask.shape
    for j,i in np.argwhere(mask):
        node=(int(j),int(i))
        for dj,di in ((0,1),(1,0),(1,1),(1,-1)):
            other=(node[0]+dj,node[1]+di)
            if not (0<=other[0]<ny and 0<=other[1]<nx and mask[other]): continue
            if di and dj and (not mask[node[0],other[1]] or not mask[other[0],node[1]]): continue
            a=np.asarray([xs[node[1]],ys[node[0]]]); b=np.asarray([xs[other[1]],ys[other[0]]])
            if capsule_transition_clear(mask,xs,ys,a,b,length):
                neighbours.setdefault(node,[]).append(other); neighbours.setdefault(other,[]).append(node)
    seen=set(); best=[]
    for seed in neighbours:
        if seed in seen: continue
        stack=[seed]; seen.add(seed); component=[]
        while stack:
            node=stack.pop(); component.append(node)
            for other in neighbours[node]:
                if other not in seen: seen.add(other); stack.append(other)
        if len(component)>len(best): best=component
    reachable=np.zeros_like(mask)
    for node in best: reachable[node]=True
    return reachable


def nearest(mask,target,xs,ys):
    cells=np.argwhere(mask); d=(xs[cells[:,1]]-target[0])**2+(ys[cells[:,0]]-target[1])**2; j,i=cells[int(np.argmin(d))]; return int(j),int(i)


def compress(points):
    keep=[0]
    for k in range(1,len(points)-1):
        a=points[k]-points[keep[-1]]; b=points[k+1]-points[k]
        if abs(a[0]*b[1]-a[1]*b[0])>1e-8: keep.append(k)
    keep.append(len(points)-1); return points[keep]


def resample_route(points,spacing=.75):
    """Densify polyline waypoints so heading changes before obstacle corners."""
    output=[np.asarray(points[0],float)]
    for a,b in zip(points,np.asarray(points)[1:]):
        distance=float(np.linalg.norm(b-a)); steps=max(1,int(math.ceil(distance/spacing)))
        output.extend(a+(b-a)*(k/steps) for k in range(1,steps+1))
    return np.asarray(output,float)


def plan_routes(masks,xs,ys,specs):
    common=np.logical_or.reduce(masks); cells=np.argwhere(common); j0,i0=cells.min(0); j1,i1=cells.max(0); cj,ci=cells.mean(0)
    # Five distinct entrance-to-exit tasks.  Four cross the junction in the
    # cardinal directions and the fifth turns from the west entrance to the
    # north exit, so their paths genuinely interact near the junction centre.
    targets=[
        ((cj-10,i0+3),(cj+7,i1-2)),       # west -> east
        ((cj+10,i1-2),(cj-7,i0+3)),       # east -> west
        ((j0+2,ci-22),(j1-3,ci+15)),      # south -> north
        ((j1-5,ci+25),(j0+4,ci-5)),       # north -> south
        ((cj+14,i0+6),(j1-8,ci+20)),      # west -> north turn
    ]
    routes=[]
    for mask,spec,(start_t,goal_t) in zip(masks,specs,targets):
        # The raster mask is used only to select reachable endpoint regions.
        # Exact static-OBB capsule checks remain on every A* edge below.
        reachable=capsule_reachable_mask(mask,xs,ys,spec["length"])
        s=nearest(reachable,(xs[int(np.clip(start_t[1],i0,i1))],ys[int(np.clip(start_t[0],j0,j1))]),xs,ys)
        g=nearest(reachable,(xs[int(np.clip(goal_t[1],i0,i1))],ys[int(np.clip(goal_t[0],j0,j1))]),xs,ys)
        cells_path=astar_capsule(mask,s,g,xs,ys,spec["length"])
        if len(cells_path)<8: raise RuntimeError(f"A* route failed {s}->{g}")
        routes.append(resample_route(compress(np.asarray([[xs[i],ys[j]] for j,i in cells_path],float))))
    # Vehicles that finish at an entrance are removed from the active road in
    # the task demo; the final pose is still recorded in metadata. This mirrors
    # an agent leaving the finite simulated map instead of becoming a new
    # permanent obstacle at its destination.
    return routes


def annotate(frame,t,agents,metrics):
    image=Image.fromarray(frame); draw=ImageDraw.Draw(image,"RGBA"); draw.rounded_rectangle((18,18,790,128),radius=10,fill=(8,12,18,178))
    counts={status:sum(a["status"]==status for a in agents) for status in ("scheduled","moving","arrived","failed")}
    draw.text((32,28),"scene_10 · 5 mixed routes · endpoint-terminated traffic task",fill="white")
    draw.text((32,54),f"t={t:05.1f}s  moving={counts['moving']}  arrived={counts['arrived']}  failed={counts['failed']}  scheduled={counts['scheduled']}",fill=(130,220,255))
    draw.text((32,80),"status: "+"  ".join(f"V{a['id']+1}={a['status'].upper()}" for a in agents),fill=(145,242,173))
    draw.text((32,105),"recording ends only after every vehicle ARRIVED or FAILED",fill=(255,211,128))
    return np.asarray(image)


def main():
    args=args_parse(); run=args.run_dir.resolve(); meta=run/"metadata"; caps=run/"captures"; vis=run/"visualizations"
    for p in (meta,caps,vis): p.mkdir(parents=True,exist_ok=False)
    source=args.usd.resolve(); tar=args.tar.resolve(); asset=args.vehicle_asset.resolve(); registry=args.registry.resolve(); audit=args.audit_inventory.resolve()
    registry_data=json.loads(registry.read_text())
    selected_records=[next(x for x in registry_data["records"] if x["asset_id"]==asset_id) for asset_id in MIXED_VEHICLE_IDS]
    mixed_assets={record["asset_id"]:Path(record["source_asset"]).resolve() for record in selected_records}
    hashes={"usd":sha256(source),"tar":sha256(tar),"legacy_asset_argument":sha256(asset),"registry":sha256(registry),"audit":sha256(audit),"mixed_vehicle_assets":{asset_id:sha256(path) for asset_id,path in mixed_assets.items()}}
    wrapper=run/"scene10_multiroute_wrapper.usda"; wrapper.write_text(f'#usda 1.0\n(\n subLayers=[@{source}@]\n)\n')
    env={"timestamp":now(),"git_commit":command_output(["git","rev-parse","HEAD"]),"git_status_short":command_output(["git","status","--short"]),"command_line":sys.argv,"physical_gpu_index":args.gpu,"nvidia_smi":command_output(["nvidia-smi"]),"isaac_sim":package_version("isaacsim"),"renderer":"RayTracedLighting","resolution":[args.width,args.height],"camera":"/MultiRouteTraffic/Camera","source_asset_hashes":hashes}
    write_json(meta/"environment.json",env); write_json(meta/"summary.json",{"status":"running","started_at":now()})
    app=timeline=annotator=render_product=None; started=time.perf_counter()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({"headless":True,"renderer":"RayTracedLighting","width":args.width,"height":args.height,"active_gpu":args.gpu,"physics_gpu":args.gpu,"multi_gpu":False,"max_gpu_count":1,"create_new_stage":False,"extra_args":["--ext-folder",str(args.ext_folder.resolve()),"--/renderer/multiGpu/enabled=false","--/app/window/hideUi=1"]},experience=str(args.experience.resolve()))
        import carb,cv2,omni.kit.app,omni.physx,omni.replicator.core as rep,omni.timeline,omni.usd
        from pxr import Gf,UsdGeom
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.asset_converter",True)
        for _ in range(4): app.update()
        context=omni.usd.get_context(); context.open_stage(str(wrapper)); wait_stage(context,app); stage=context.get_stage(); stage.SetEditTarget(stage.GetRootLayer())
        carb.settings.get_settings().set("/rtx/post/tonemap/exposure",0.0); timeline=omni.timeline.get_timeline_interface(); timeline.play()
        for _ in range(24): app.update()
        lane=LaneFootprint(stage); static=static_vehicles(stage,audit)
        moving_paths={record["scene_prim_path"] for record in selected_records}
        static=[item for item in static if item[0] not in moving_paths]
        write_json(meta/"static_vehicle_bodies.json",[{"path":path,"corners_xy":poly.tolist()} for path,poly in static])
        records=selected_records
        specs=[]
        for rec in records:
            if not rec["eligibility"]["traffic_ready"] or rec["front_direction"]["status"]=="pending_visual_validation":
                raise RuntimeError(f"vehicle is not traffic/front ready: {rec['asset_id']}")
            length,width=map(float,rec["footprint"]["length_width_m"])
            specs.append({"asset_id":rec["asset_id"],"category":rec["category"],"asset":Path(rec["source_asset"]),"length":length,"width":width,"center_height":float(rec["grounding"]["body_center_height_above_support_m"]),"radius":.5*math.hypot(length,width)+.18})
        xs,ys,road_clearance,obstacle_clearance=build_clearance_fields(lane,static)
        grids=[]
        for spec in specs:
            grids.append(build_grid(road_clearance,obstacle_clearance,spec["width"]/2))
        np.savez_compressed(meta/"lane_vehicle_occupancy.npz",xs=xs,ys=ys,valid=np.logical_or.reduce(grids),**{f"valid_{i}":grid for i,grid in enumerate(grids)})
        routes=plan_routes(grids,xs,ys,specs)
        route_lengths=[float(np.linalg.norm(np.diff(r,axis=0),axis=1).sum()) for r in routes]
        agents=[]
        route_names=("west_to_east","east_to_west","south_to_north","north_to_south","west_to_north_turn")
        junction_center=np.asarray([-625.,490.]); desired_merge_times=(60.,14.,22.,32.,95.)
        for i,(route,spec,route_name) in enumerate(zip(routes,specs,route_names)):
            speed=3.2-.08*i; nearest_center=int(np.argmin(np.linalg.norm(route-junction_center,axis=1)))
            distance_to_merge=float(np.linalg.norm(np.diff(route[:nearest_center+1],axis=0),axis=1).sum())
            # Vehicles enter the same conflict area in a controlled sequence.
            # The opposing east->west car clears before the westbound pair is
            # released, while the two crossing cars still exercise reciprocal
            # yielding.  The west->north turn follows the west->east car with
            # a real traffic headway instead of sharing its circular ORCA zone.
            start_time=max(0.,desired_merge_times[i]-distance_to_merge/speed)
            d=route[1]-route[0]; agents.append({"id":i,"kind":"car","route_name":route_name,"asset_id":spec["asset_id"],"category":spec["category"],"length":spec["length"],"width":spec["width"],"radius":spec["radius"],"preferred_speed":speed,"position":route[0].copy(),"velocity":np.zeros(2),"heading":math.atan2(d[1],d[0]),"route":route,"route_locked_velocity":True,"waypoint":1,"direction":1,"goal_reversals":0,"trajectory":[route[0].copy()],"status":"scheduled","start_time_s":start_time,"last_progress_time_s":start_time,"best_goal_distance_m":float(np.linalg.norm(route[-1]-route[0])),"terminal_time_s":None})
        # Scheduled vehicles are inactive and hidden until their release time,
        # so entrance positions may be shared without creating a scene overlap.
        write_json(meta/"routes.json",{"route_lengths_m":route_lengths,"routes":[r.tolist() for r in routes],"vehicles":[{k:(str(v) if isinstance(v,Path) else v) for k,v in s.items()} for s in specs],"planning_radius_m":None,"per_vehicle_lateral_planning_radius_m":[s["width"]/2+STATIC_SAFETY_MARGIN_M for s in specs],"planner":"per-vehicle lateral occupancy + rectangular/capsule swept-edge 8-neighbor A*"})
        if args.preflight_only:
            write_json(meta/"summary.json",{"status":"preflight_passed","static_vehicle_count":len(static),"route_lengths_m":route_lengths,"occupancy_cells_by_vehicle":[int(grid.sum()) for grid in grids]}); return 0
        vehicles=[NativeVehicle(stage,record) for record in records]
        query=omni.physx.get_physx_scene_query_interface(); target=Gf.Vec3d(-625.,490.,1.); eye=target+Gf.Vec3d(30.,-30.,120.)
        camera=UsdGeom.Camera.Define(stage,"/MultiRouteTraffic/Camera"); camera.CreateFocalLengthAttr(24.); camera.CreateHorizontalApertureAttr(24.); camera.CreateClippingRangeAttr(Gf.Vec2f(.1,100000.)); xf=UsdGeom.Xformable(camera.GetPrim()); xf.AddTransformOp().Set(Gf.Matrix4d(1).SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse())
        render_product=rep.create.render_product(camera.GetPath(),(args.width,args.height),force_new=True); annotator=rep.AnnotatorRegistry.get_annotator("rgb"); annotator.attach([render_product]); [app.update() for _ in range(30)]
        max_count=args.frames or int(args.duration*args.fps); writer=cv2.VideoWriter(str(caps/"scene10_multiroute_orca_traffic.webm"),cv2.VideoWriter_fourcc(*"VP90"),args.fps,(args.width,args.height)); keys=[]; states=[]
        min_dynamic=math.inf; min_static=math.inf; overlaps=0; static_hits=0; lane_rejects=0; constraints=0; conflicts=0; max_rendered_center_error=0.; rendered_center_offsets=[None]*len(agents); query_dt=1/(args.fps*4)
        try:
            for frame_idx in range(max_count):
                t=frame_idx/args.fps
                for a in agents:
                    if a["status"]=="scheduled" and t+1e-9>=a["start_time_s"]: a["status"]="moving"; a["last_progress_time_s"]=t
                metrics={"orca_constraint_count":0,"predicted_conflicts":0}
                if frame_idx:
                    for _ in range(4):
                        active=[a for a in agents if a["status"]=="moving"]
                        if not active: break
                        before=[{
                            "position":a["position"].copy(),
                            "velocity":a["velocity"].copy(),
                            "heading":float(a["heading"]),
                            "waypoint":int(a["waypoint"]),
                            "direction":int(a["direction"]),
                            "goal_reversals":int(a["goal_reversals"]),
                            "trajectory_length":len(a["trajectory"]),
                        } for a in active]
                        row=step_orca(active,query_dt,time_horizon=4.,neighbor_distance=14.,safety_margin=.3)
                        metrics["orca_constraint_count"]+=row["orca_constraint_count"]; metrics["predicted_conflicts"]+=row["predicted_conflicts"]
                        for a,old in zip(active,before):
                            poly=footprint(a["position"],a["heading"],a["length"],a["width"])
                            if not all(lane.contains(c) for c in poly) or any(sat_intersects(poly,p,margin=STATIC_SAFETY_MARGIN_M) for _,p in static):
                                # ORCA updates position, heading and route progress as one
                                # state transition. Restore all of them together; restoring
                                # only XY allowed a stopped car to rotate into an obstacle.
                                a["position"]=old["position"]
                                a["velocity"]=np.zeros(2)
                                a["heading"]=old["heading"]
                                a["waypoint"]=old["waypoint"]
                                a["direction"]=old["direction"]
                                a["goal_reversals"]=old["goal_reversals"]
                                del a["trajectory"][old["trajectory_length"]:]
                                a["trajectory"].append(a["position"].copy())
                                lane_rejects+=1
                    for a in agents:
                        if a["status"]!="moving": a["velocity"]=np.zeros(2); continue
                        goal_distance=float(np.linalg.norm(a["route"][-1]-a["position"]))
                        if goal_distance<=args.arrival_radius or a["goal_reversals"]>0:
                            a["position"]=a["route"][-1].copy(); a["velocity"]=np.zeros(2); a["status"]="arrived"; a["terminal_time_s"]=t; continue
                        if goal_distance<a["best_goal_distance_m"]-.20:
                            a["best_goal_distance_m"]=goal_distance; a["last_progress_time_s"]=t
                        elif t-a["start_time_s"]>=120.0:
                            a["velocity"]=np.zeros(2); a["status"]="failed"; a["terminal_time_s"]=t
                        elif t-a["last_progress_time_s"]>=args.stall_timeout:
                            a["velocity"]=np.zeros(2); a["status"]="failed"; a["terminal_time_s"]=t
                constraints+=int(metrics["orca_constraint_count"]); conflicts+=int(metrics["predicted_conflicts"]); polys=[]; frame_state=[]
                for a,v in zip(agents,vehicles):
                    p=a["position"]; ground,ground_path=physical_road_height(query,p); center,poly,_=v.set_pose(p,float(a["heading"]),ground)
                    active_on_road=a["status"]=="moving"
                    if active_on_road: polys.append(poly)
                    frame_state.append({"id":a["id"],"asset_id":a["asset_id"],"category":a["category"],"status":a["status"],"active_on_road":active_on_road,"center_xyz":center.tolist(),"goal_xy":a["route"][-1].tolist(),"goal_distance_m":float(np.linalg.norm(a["route"][-1]-p)),"ground_z":ground,"ground_source":"per-frame downward PhysX raycast","ground_collision":ground_path,"velocity":a["velocity"].tolist(),"heading_deg":math.degrees(a["heading"]),"waypoint":a["waypoint"],"goal_reversals":a["goal_reversals"]})
                    v.prim.SetActive(active_on_road)
                for vehicle_index,(state,v) in enumerate(zip(frame_state,vehicles)):
                    actual_world=np.asarray(UsdGeom.Xformable(v.prim).ComputeLocalToWorldTransform(0),float).T
                    measured=(actual_world@np.r_[v.local_body_center,1.0])[:3]
                    raw_offset=measured[:2]-np.asarray(state["center_xyz"][:2],float)
                    if rendered_center_offsets[vehicle_index] is None: rendered_center_offsets[vehicle_index]=raw_offset.copy()
                    error=float(np.linalg.norm(raw_offset-rendered_center_offsets[vehicle_index]))
                    state["rendered_center_xyz"]=measured.tolist(); state["rendered_center_xy_error_m"]=error
                    max_rendered_center_error=max(max_rendered_center_error,error)
                for i,p in enumerate(polys):
                    for q in polys[i+1:]: min_dynamic=min(min_dynamic,polygon_clearance(p,q)); overlaps+=int(sat_intersects(p,q))
                    for _,q in static: min_static=min(min_static,polygon_clearance(p,q)); static_hits+=int(sat_intersects(p,q))
                rep.orchestrator.step(rt_subframes=1); frame=annotate(rgb_array(annotator),t,agents,metrics); writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)); states.append({"frame":frame_idx,"time_s":t,"agents":frame_state})
                if frame_idx==0 or frame_idx%int(20*args.fps)==0: path=caps/f"frame_{frame_idx:04d}.png"; Image.fromarray(frame).save(path); keys.append(path)
                if all(a["status"] in ("arrived","failed") for a in agents):
                    path=caps/f"frame_{frame_idx:04d}.png"; Image.fromarray(frame).save(path)
                    if path not in keys: keys.append(path)
                    break
        finally: writer.release()
        count=len(states)
        write_json(meta/"traffic_states.json",states); contact_sheet(keys,vis/"multiroute_contact_sheet.png")
        cap=cv2.VideoCapture(str(caps/"scene10_multiroute_orca_traffic.webm")); video={"opened":bool(cap.isOpened()),"frame_count":int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),"width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),"height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),"fps":cap.get(cv2.CAP_PROP_FPS)}; cap.release()
        terminal=all(a["status"] in ("arrived","failed") for a in agents)
        all_arrived=all(a["status"]=="arrived" for a in agents)
        passed=all_arrived and overlaps==0 and static_hits==0 and max_rendered_center_error<=.10 and video["frame_count"]==count
        summary={"status":"passed" if passed else "failed_validation","termination_reason":"all_vehicles_terminal" if terminal else "maximum_duration_reached","all_vehicles_arrived":all_arrived,"recorded_duration_s":count/args.fps,"maximum_duration_s":args.duration,"arrival_radius_m":args.arrival_radius,"stall_timeout_s":args.stall_timeout,"vehicle_outcomes":[{"id":a["id"],"route_name":a["route_name"],"status":a["status"],"start_xy":a["route"][0].tolist(),"goal_xy":a["route"][-1].tolist(),"terminal_time_s":a["terminal_time_s"],"final_goal_distance_m":float(np.linalg.norm(a["route"][-1]-a["position"]))} for a in agents],"dynamic_vehicle_count":len(agents),"dynamic_vehicle_models":[{"asset_id":a["asset_id"],"category":a["category"]} for a in agents],"static_vehicle_count":len(static),"static_safety_margin_m":STATIC_SAFETY_MARGIN_M,"route_lengths_m":route_lengths,"occupancy_cells_by_vehicle":[int(grid.sum()) for grid in grids],"per_vehicle_lateral_planning_radius_m":[s["width"]/2+STATIC_SAFETY_MARGIN_M for s in specs],"maximum_rendered_center_relative_xy_error_m":max_rendered_center_error,"rendered_center_relative_xy_error_limit_m":.10,"rendered_center_initial_offsets_m":[offset.tolist() for offset in rendered_center_offsets],"minimum_dynamic_obb_clearance_m":min_dynamic,"minimum_static_obb_clearance_m":min_static,"dynamic_obb_overlap_events":overlaps,"static_obb_overlap_events":static_hits,"boundary_or_static_step_rejections":lane_rejects,"orca_constraint_count":constraints,"orca_predicted_conflict_count":conflicts,"video_validation":video,"ground_height_source":"per-frame downward PhysX raycast against physical road surface; semantic Lane is XY-only","method":"fixed start/goal task + staggered release + per-vehicle lateral Lane occupancy + capsule swept-edge A* + physical-road grounding + CPU disc ORCA + explicit ARRIVED/FAILED termination","limitations":["CPU implementation, not authors' GPU ORCA","kinematic whole-body vehicles; no wheel dynamics"]}
        hashes_after={"usd":sha256(source),"tar":sha256(tar),"legacy_asset_argument":sha256(asset),"registry":sha256(registry),"audit":sha256(audit),"mixed_vehicle_assets":{asset_id:sha256(path) for asset_id,path in mixed_assets.items()}}
        if hashes_after != hashes: raise RuntimeError("source hash changed during render")
        summary["source_modified"]=False
        write_json(meta/"summary.json",summary); print(json.dumps(summary,indent=2)); return 0 if passed else 2
    except Exception as e:
        write_json(meta/"summary.json",{"status":"failed","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}); print(traceback.format_exc(),file=sys.stderr); return 1
    finally:
        if timeline: timeline.stop()
        if annotator: annotator.detach()
        if app: app.close()


if __name__=="__main__": raise SystemExit(main())
