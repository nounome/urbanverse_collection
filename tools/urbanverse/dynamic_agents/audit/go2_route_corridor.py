"""Live PhysX ground/corridor check, ignoring only own robot and added support.

Source-scene collisions remain enabled. A support plane cannot make a missing
source ground surface pass this admission test.
"""
import math
import numpy as np


def robot_collision_footprint(robot):
    """Measure posed collision geometry in the base frame, including every foot."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    stage=omni.usd.get_context().get_stage(); cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'])
    # Kit conversion/render updates can advance PhysX independently of an
    # Isaac Lab env step. Refresh timestamped buffers, then use the base link
    # from the SAME body-state snapshot rather than a differently aged root.
    robot.update(1.e-6)
    xyz=robot.data.body_pos_w[0].cpu().numpy(); quat=robot.data.body_quat_w[0].cpu().numpy()
    base_index=robot.body_names.index('base')
    base_xyz=xyz[base_index];base_q=quat[base_index]
    inverse=Rotation.from_quat(base_q[[1,2,3,0]]).inv(); all_points=[]; per_body=[]
    for i,name in enumerate(robot.body_names):
        body=stage.GetPrimAtPath('/World/envs/env_0/Robot/'+name)
        if not body.IsValid(): continue
        corners=[]
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            box=cache.ComputeRelativeBound(prim,body).ComputeAlignedRange()
            if box.IsEmpty():continue
            lo,hi=np.array(box.GetMin()),np.array(box.GetMax())
            corners.extend([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        if corners:
            world=Rotation.from_quat(quat[i,[1,2,3,0]]).apply(corners)+xyz[i]
            local=inverse.apply(world-base_xyz);all_points.extend(local.tolist());per_body.append(name)
    if not all_points:raise RuntimeError('No robot collision geometry found')
    a=np.array(all_points)
    return dict(width_m=float(np.ptp(a[:,1])),length_m=float(np.ptp(a[:,0])),
                half_width_m=float(np.max(np.abs(a[:,1]))),bodies=per_body,
                bounds_base_frame=[a.min(0).tolist(),a.max(0).tolist()],scope='nominal posed collision envelope; dynamic foot swing still requires test')


def audit_route_corridor(route):
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()
    spec=route['live_route_preflight']; points=np.array(route['points_xy'])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    ss=np.arange(0,arc[-1],spec.get('spacing_m',.15)); radius=spec.get('radius_m',.55)
    ground=route['ground_z']; samples=[]; failed=0
    def ignored(path): return '/Robot/' in path or 'Go2RuntimeSupport' in path
    for s in ss:
        idx=min(np.searchsorted(arc,s,side='right')-1,len(points)-2)
        tangent=points[idx+1]-points[idx];tangent/=np.linalg.norm(tangent)
        p=points[idx]+(s-arc[idx])*tangent;normal=np.array([-tangent[1],tangent[0]])
        for offset in [-radius,0.,radius]:
            xy=p+offset*normal; hits=[]
            def report(hit):
                path=str(hit.rigid_body)
                if not ignored(path): hits.append(dict(path=path,z=float(hit.position[2]),normal=list(hit.normal)))
                return True
            query.raycast_all((float(xy[0]),float(xy[1]),ground+5.),(0.,0.,-1.),8.,report,True)
            heights=[h for h in hits if h['z']>=ground-1.]
            top=max(heights,key=lambda h:h['z']) if heights else None
            ok=top is not None and abs(top['z']-ground)<=spec.get('maximum_ground_step_m',.22)
            failed+=not ok
            samples.append(dict(arc_m=float(s),offset_m=offset,xy=xy.tolist(),passed=bool(ok),highest_source_hit=top))
    return dict(passed=failed==0,sample_count=len(samples),failed_samples=int(failed),source_ground_required=True,
                maximum_ground_step_m=spec.get('maximum_ground_step_m',.22),radius_m=radius,samples=samples)


def audit_lower_row_grid(output_dir, ground):
    """Map live source collider tops; this is evidence, not a maintained input."""
    from omni.physx import get_physx_scene_query_interface
    from PIL import Image
    query=get_physx_scene_query_interface()
    xs=np.arange(-688.,-658.,.04); ys=np.arange(484.8,490.4,.04)
    heights=np.full((len(ys),len(xs)),np.nan,dtype=np.float32)
    for row,y in enumerate(ys):
        for col,x in enumerate(xs):
            hits=[]
            def report(hit):
                path=str(hit.rigid_body)
                if '/Robot/' not in path and 'Go2RuntimeSupport' not in path and hit.position[2]>=ground-1:
                    hits.append(float(hit.position[2]))
                return True
            query.raycast_all((float(x),float(y),ground+5),(0.,0.,-1.),8.,report,True)
            if hits: heights[row,col]=max(hits)
    np.savez_compressed(output_dir/'live_lower_row_grid.npz',xs=xs,ys=ys,heights=heights,ground=ground)
    rgb=np.zeros((*heights.shape,3),np.uint8);rgb[:]=[24,30,36]
    free=np.isfinite(heights)&(np.abs(heights-ground)<=.22)
    rgb[free]=[80,170,150];rgb[np.isfinite(heights)&~free]=[220,100,65]
    Image.fromarray(rgb[::-1]).resize((1500,280)).save(output_dir/'live_lower_row_grid.png')
    return dict(bounds_xy=[-688.,484.8,-658.,490.4],spacing_m=.04,shape=list(heights.shape),free_cells=int(free.sum()))
