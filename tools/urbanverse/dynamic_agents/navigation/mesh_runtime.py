"""Runtime admission and run-local USD cleanup for versioned mesh maps."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .mesh_loop_workflow import MeshMap, body_clear
from .cleanup_categories import cleanup_category


class MeshRoadFootprint:
    """Configured Lane minus live-mesh obstacles, with whole-body checks."""
    def __init__(self,inventory:Path,settings:Path):
        self.grid=MeshMap(inventory)
        self.settings=json.loads(settings.read_text())
        self.free=self.grid.lane&~self.grid.occupancy(self.settings.get('removed_instance_roots',[]))
        self.fallback_ground_z_m=float(self.grid.meta['ground_z_m'])
        self.query_count=0;self.candidate_count=0
        self.spatial_cell_size_m=float(self.grid.pixel.mean())

    def contains(self,point,tolerance=0.):
        # Never expand forbidden mesh pixels by the legacy .35 m tolerance.
        self.query_count+=1
        return bool(self.grid.contains(self.free,np.asarray(point)))

    def height(self,point,tolerance=0.):
        if not self.contains(point):raise RuntimeError('No mesh-qualified road at '+str(point))
        return self.fallback_ground_z_m

    def pose_clear(self,position,heading,length,width,margin=.15):
        return bool(body_clear(self.grid,self.free,np.asarray(position)[None,:],
            np.array([heading]),length,width,margin)[0])


def from_scene(scene):
    spec=scene.traffic.get('mesh_navigation') if scene is not None else None
    if not spec:return None
    road=MeshRoadFootprint((scene.path.parent/spec['inventory']).resolve(),
                          (scene.path.parent/spec['settings']).resolve())
    road.fallback_ground_z_m=scene.fallback_ground_z_m
    return road


def author_cleanup(wrapper:Path,scene):
    """Deactivate exact referenced roots in a new wrapper, never source assets."""
    spec=scene.traffic.get('mesh_navigation') if scene is not None else None
    if not spec:return None
    from pxr import Usd
    settings=json.loads((scene.path.parent/spec['settings']).resolve().read_text())
    inventory=json.loads((scene.path.parent/spec['inventory']).resolve().read_text())
    roots=settings.get('removed_instance_roots',[])
    limit=settings.get('maximum_cleanup_objects',6)
    if limit is not None and len(roots)>limit:raise ValueError('Cleanup budget exceeded')
    allowed={r['instance_root'] for r in inventory['obstacles'] if r['kind']!='building'}
    stage=Usd.Stage.Open(str(wrapper))
    base=stage.GetDefaultPrim().GetPath()
    rows=[]
    for root in roots:
        if root not in allowed or cleanup_category(root) is None:
            raise ValueError('Unapproved cleanup category/root: '+root)
        prim=stage.GetPrimAtPath(base.AppendChild(root))
        if not prim.IsValid():raise RuntimeError('Cleanup root absent from wrapper: '+root)
        prim.SetActive(False)
        rows.append(dict(instance_root=root,wrapper_prim=str(prim.GetPath()),
            runtime_prim='/World/ground/terrain/'+root,reason='clearance for admitted traffic route',
            visual_and_collision_deactivated=True))
    stage.GetRootLayer().Save()
    return dict(source_unchanged=True,wrapper=str(wrapper),removed=rows)


def audit_loaded_meshes(stage,scene):
    """Reject missing GLB payloads instead of navigating an empty city."""
    from pxr import UsdGeom
    spec=scene.traffic['mesh_navigation']
    inventory=json.loads((scene.path.parent/spec['inventory']).resolve().read_text())
    settings=json.loads((scene.path.parent/spec['settings']).resolve().read_text())
    removed=set(settings.get('removed_instance_roots',[]));missing=[];count=0
    for row in inventory['obstacles']:
        if row['instance_root'] in removed:continue
        path='/World/ground/terrain/'+row['source_relative_path']
        prim=stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Mesh) or not UsdGeom.Mesh(prim).GetPointsAttr().Get():
            missing.append(path)
        else:count+=1
    still_active=[]
    for root in removed:
        prim=stage.GetPrimAtPath('/World/ground/terrain/'+root)
        if prim.IsValid() and prim.IsActive():still_active.append(root)
    return dict(passed=not missing and not still_active,verified_mesh_count=count,
        missing_expected_meshes=missing,removed_roots_still_active=still_active,
        removed_instance_roots=sorted(removed),scope='live stage identity/geometry presence; not pixel validation')
