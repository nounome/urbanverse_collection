"""Run-local kinematic contact proxies; never used as visible actor substitutes."""
from __future__ import annotations
import json
from pathlib import Path
import math


def author(wrapper:Path,vehicle_registry:Path,counts,micro_catalog,road_z,walk_z):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from ..traffic.multivehicle_manager import DIVERSE_TRAFFIC_VEHICLE_IDS
    stage=Usd.Stage.Open(str(wrapper));root=str(stage.GetDefaultPrim().GetPath())+'/DynamicContactProxies'
    # Controller states are world coordinates. The referenced city's default
    # prim has its own translation; do not apply that offset a second time.
    UsdGeom.Xform.Define(stage,root).SetResetXformStack(True)
    registry=json.loads(vehicle_registry.read_text());records={r['asset_id']:r for r in registry['records']}
    rows=[]
    for i in range(counts['vehicles']):
        record=records[DIVERSE_TRAFFIC_VEHICLE_IDS[i%len(DIVERSE_TRAFFIC_VEHICLE_IDS)]]
        length,width=record['footprint']['length_width_m']
        rows.append(dict(name=f'Car_{i:02d}',kind='vehicle',id=i,size=[length,width,1.5],ground_z=road_z))
    for i in range(counts['people']):rows.append(dict(name=f'Person_{i:02d}',kind='person',id=i,size=[.6,.6,1.7],ground_z=walk_z))
    asset_ids=list(micro_catalog)
    for i in range(counts['micromobility']):
        c=micro_catalog[asset_ids[i%len(asset_ids)]]
        rows.append(dict(name=f'Micro_{i:02d}',kind='micro',id=f'TwoWheeler_{i:02d}',size=[c.length_m,c.width_m,c.height_m],ground_z=walk_z))
    group=UsdPhysics.CollisionGroup.Define(stage,root+'/Filter')
    group.CreateFilteredGroupsRel().AddTarget(group.GetPath())
    included=[]
    for i,row in enumerate(rows):
        cube=UsdGeom.Cube.Define(stage,root+'/'+row['name']);cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.,0.,-500.-3*i));cube.AddRotateZOp().Set(0.)
        cube.AddScaleOp().Set(Gf.Vec3f(*row['size']));cube.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        body=UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim());body.CreateKinematicEnabledAttr(True)
        included.append(cube.GetPath())
        row['runtime_path']='/World/ground/terrain/DynamicContactProxies/'+row['name']
    group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(included)
    stage.GetRootLayer().Save()
    return dict(rows=rows,shape='conservative boxes, not exact visible mesh',
        motion='USD-driven kinematic boxes; verified by PhysX queries rather than stale GPU kinematic tensor reads',
        limitation='discrete contact geometry, not calibrated impact dynamics',
        proxy_proxy_collision_filtered=True,go2_collision_enabled=True,
        physical_contact_verified=False)


class Runtime:
    def __init__(self,stage,metadata):
        from pxr import UsdGeom
        self.stage=stage;self.rows=metadata['rows'];self.ops=[];self.targets={}
        for row in self.rows:
            prim=stage.GetPrimAtPath(row['runtime_path'])
            if not prim.IsValid():raise RuntimeError('Contact proxy missing: '+row['runtime_path'])
            self.ops.append(UsdGeom.Xformable(prim).GetOrderedXformOps()[:2])

    def update(self,vehicles,people,micro):
        from pxr import Gf
        for row,(translate,rotate) in zip(self.rows,self.ops):
            if row['kind']=='vehicle':
                if row['id']>=len(vehicles):continue
                actor=vehicles[row['id']]
                if actor['status'] not in ('moving','stopped'):continue
                x,y=actor['position'];yaw=actor['heading'];z=row['ground_z']
            elif row['kind']=='person':
                if row['id']>=len(people):continue
                x,y,z=people[row['id']];yaw=0.
            else:
                if row['id'] not in micro:continue
                actor=micro[row['id']];x,y=actor.position_xy;yaw=actor.yaw_rad;z=row['ground_z']
            translate.Set(Gf.Vec3d(float(x),float(y),float(z+row['size'][2]/2)))
            rotate.Set(float(math.degrees(yaw)))
            self.targets[row['runtime_path']]=(float(x),float(y),float(z),float(row['size'][2]))

    def audit_physx(self,scene_query):
        """Down-rays verify real PhysX query shapes, not merely USD attributes."""
        import carb
        from pxr import Usd,UsdGeom,UsdPhysics
        cache=UsdGeom.XformCache()
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'],True,True)
        rows=[]
        for path,(x,y,z,height) in self.targets.items():
            hit=scene_query.raycast_closest(carb.Float3(x,y,z+height+.2),carb.Float3(0.,0.,-1.),height+.4)
            collision=str(hit.get('collision',''))
            collisions=[]
            def collect(query_hit):
                collisions.append(str(query_hit.collision))
                return True
            scene_query.raycast_all(carb.Float3(x,y,z+height+.2),carb.Float3(0.,0.,-1.),height+.4,collect)
            prim=self.stage.GetPrimAtPath(path)
            box=bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            rows.append(dict(expected_path=path,hit=bool(hit.get('hit')),actual_collision=collision,
                usd_world_translation=list(cache.GetLocalToWorldTransform(prim).ExtractTranslation()),
                usd_world_bounds=[list(box.GetMin()),list(box.GetMax())],
                collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                simulation_owners=[str(v) for v in UsdPhysics.RigidBodyAPI(prim).GetSimulationOwnerRel().GetTargets()],
                all_collisions=collisions,
                passed=path in collisions))
        return dict(passed=bool(rows) and all(r['passed'] for r in rows),rows=rows,
            physics_scenes=[str(p.GetPath()) for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)],
            physics_settings={key:carb.settings.get_settings().get('/physics/'+key)
                for key in ['updateFromUsd','updateToUsd','fabricUpdateTransformations','suppressReadback']},
            scope='PhysX query proxy presence at current targets; not robot contact or pixel evidence')
