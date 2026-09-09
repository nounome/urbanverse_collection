"""Seeded A* + smoothing on the same live-mesh free map used by traffic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from urbanverse.dynamic_agents.navigation.mesh_loop_workflow import MeshMap,digest
from urbanverse.dynamic_agents.navigation.automation_rules import select_distance_route
from urbanverse.dynamic_agents.navigation.global_route_planner import astar
from urbanverse.dynamic_agents.review.build_go2_complex_route_preview import rounded_polyline,resample,route_metrics


def conservative_collider_mask(grid, roots):
    """Optional Go2-only exclusions for unresolved mesh/PhysX discrepancies.

    Use the complete mesh bounds of explicitly identified instances, including
    overhead geometry. This adds occupancy; it never removes source collisions
    or changes the common traffic map. Clearance dilation is applied downstream.
    """
    roots=set(roots)
    mask=np.zeros((grid.h,grid.w),np.uint8)
    found=set()
    for row in grid.obstacles:
        if row['instance_root'] not in roots:continue
        bounds=np.asarray(row['bounds'],dtype=float)
        if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or (bounds[1]<bounds[0]).any():
            raise ValueError('Invalid collider exclusion bounds: '+row['path'])
        lo,hi=bounds
        poly=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]]
        cv2.fillPoly(mask,[np.floor(grid.pixels(poly)).astype(np.int32)],1)
        found.add(row['instance_root'])
    if roots-found:raise ValueError('Unknown collider exclusion roots: '+str(sorted(roots-found)))
    return mask.astype(bool)


def generate(inventory,settings_path,output,automation_defaults=None):
    if output.exists():raise FileExistsError(output)
    settings=json.loads(settings_path.read_text());spec=settings['go2_route'];grid=MeshMap(inventory)
    if automation_defaults:
        defaults=json.loads(Path(automation_defaults).read_text())
        spec.setdefault('target_distances_m',defaults['go2_target_distances_m'])
    occupied=grid.occupancy(settings.get('removed_instance_roots',[]))
    occupied |= conservative_collider_mask(grid,spec.get('conservative_collider_roots',[]))
    domain=grid.lane if spec.get('domain','Lane')=='Lane' else (grid.lane|grid.shared|(grid.semantic==2))
    bounds=spec.get('bounds_xy',grid.view);pix=grid.pixels([[bounds[0],bounds[3]],[bounds[2],bounds[1]]]).astype(int)
    pix[:,0]=np.clip(pix[:,0],0,grid.w);pix[:,1]=np.clip(pix[:,1],0,grid.h)
    x0,y0=pix[0];x1,y1=pix[1]
    if x1-x0<3 or y1-y0<3:raise ValueError('Go2 search bounds contain no usable map interior')
    free=(domain&~occupied)[y0:y1,x0:x1].copy();free[[0,-1]]=False;free[:,[0,-1]]=False
    distance=distance_transform_edt(free,sampling=grid.pixel[::-1])
    valid=distance>=spec['clearance_radius_m']+spec.get('planning_extra_clearance_m',.08)
    rng=np.random.default_rng(spec['seed']);best=None
    _,labels,stats,_=cv2.connectedComponentsWithStats(valid.astype(np.uint8),connectivity=8)
    ranked=sorted(range(1,len(stats)),key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)[:8]
    pools=[np.argwhere(labels==i) for i in ranked if stats[i,cv2.CC_STAT_AREA]>100]
    if not pools:raise RuntimeError('No clearance-qualified Go2 component')
    qualified=[]
    rejects=dict(incomplete_goals=0,astar=0,outside=0,occupied=0,curvature=0)
    for attempt in range(spec.get('candidate_count',48)):
        # Sample within an actual connected component's extent, not padded
        # image bands (which may contain no road). Try several components.
        cells=pools[attempt%len(pools)]
        span=np.ptp(cells,axis=0)*grid.pixel[::-1]
        axis=int(np.argmax(span))
        if spec.get('target_distances_m'):
            # Keep searches proportionate to the requested distance instead of
            # spanning a kilometre-scale domain and discarding most of the path.
            window=1.5*max(spec['target_distances_m'])/grid.pixel[::-1][axis]
            lo,hi=cells[:,axis].min(),cells[:,axis].max()
            if hi-lo>window:
                start=rng.uniform(lo,hi-window)
                cells=cells[(cells[:,axis]>=start)&(cells[:,axis]<=start+window)]
            # Avoid sampling alternating sides of a large building. A local
            # transverse strip still permits bends, without arbitrary scene XY.
            other=1-axis
            half_width=spec.get('sampling_strip_width_m',35.)/(2*grid.pixel[::-1][other])
            centre=cells[rng.integers(len(cells)),other]
            cells=cells[np.abs(cells[:,other]-centre)<=half_width]
        bands=np.linspace(cells[:,axis].min(),cells[:,axis].max()+1,spec['goal_count']+1)
        goals=[]
        for low,high in zip(bands[:-1],bands[1:]):
            pool=cells[(cells[:,axis]>=low+3)&(cells[:,axis]<high-3)]
            if not len(pool):break
            goals.append(pool[rng.integers(len(pool))])
        if len(goals)!=spec['goal_count']:
            rejects['incomplete_goals']+=1
            continue
        goals=goals[::-1]  # reverse ordered bands along the map's longer axis
        path=[]
        for a,b in zip(goals[:-1],goals[1:]):
            leg=astar(tuple(a),tuple(b),valid,distance,2.,float(grid.pixel.mean()),.9,1.,
                      maximum_expansions=spec.get('maximum_astar_expansions',100000))
            if leg is None:
                rejects['astar']+=1
                break
            path.extend(leg if not path else leg[1:])
        else:
            xy=grid.world(np.array([(x+x0,y+y0) for y,x in path]))
            controls=cv2.approxPolyDP(xy.astype(np.float32),.25,False).reshape(-1,2)
            for smoothing in spec.get('smoothing_radii_m',(6.,4.,2.5,1.5,.9,.6,.4)):
                dense=rounded_polyline(controls,smoothing,.04)
                p=np.floor(grid.pixels(dense)).astype(int)-[x0,y0]
                if not ((p[:,0]>=0)&(p[:,0]<valid.shape[1])&(p[:,1]>=0)&(p[:,1]<valid.shape[0])).all():
                    rejects['outside']+=1
                    continue
                if not valid[p[:,1],p[:,0]].all():
                    rejects['occupied']+=1
                    continue
                metrics=route_metrics(dense,controls)
                if metrics['max_sampled_curvature_rad_per_stage_unit']>2.8:
                    rejects['curvature']+=1
                    continue
                qualified.append(dense)
                length=float(np.linalg.norm(np.diff(dense,axis=0),axis=1).sum())
                yaw=np.unwrap(np.arctan2(np.diff(dense,axis=0)[:,1],np.diff(dense,axis=0)[:,0]))
                score=float(np.abs(np.diff(yaw)).sum())+length*.1
                if best is None or score>best[0]:best=(score,dense,controls,metrics,attempt)
    if best is None:raise RuntimeError('No static-free smoothed Go2 route found: '+json.dumps(rejects))
    _,dense,controls,metrics,attempt=best
    distance_selection=None
    if spec.get('target_distances_m'):
        dense,distance_selection=select_distance_route(qualified,spec['target_distances_m'],
            spec.get('distance_tolerance_fraction',.1),spec.get('maximum_repeat_ratio',2.5))
        controls=cv2.approxPolyDP(dense.astype(np.float32),.25,False).reshape(-1,2)
        metrics=route_metrics(dense,controls)
        p=np.floor(grid.pixels(dense)).astype(int)-[x0,y0]
        if not valid[p[:,1],p[:,0]].all() or metrics['max_sampled_curvature_rad_per_stage_unit']>2.8:
            raise ValueError('Distance-selected prefix failed final geometry check')
        attempt=None  # selection indexes the qualified pool, not an original attempt
    result=dict(schema_version=1,ground_z=spec.get('ground_z_m',grid.meta['ground_z_m']),points_xy=resample(dense,.1).tolist(),
        control_points_xy=controls.tolist(),metrics=metrics,
        runtime_physics_support=dict(enabled=True,corridor_width_m=1.5,endpoint_padding_m=1.,thickness_m=.1),
        live_route_preflight=dict(enabled=True,spacing_m=.1,radius_m=spec['clearance_radius_m'],maximum_ground_step_m=.22),
        generation=dict(method='seeded band goals + A* + Bezier smoothing on authoritative mesh mask',
            seed=spec['seed'],candidate_index=attempt,inventory_sha256=digest(inventory),settings_sha256=digest(settings_path),
            removed_instance_roots=settings.get('removed_instance_roots',[]),distance_selection=distance_selection,
            effective_search_bounds_xy=list(map(float,bounds)),
            effective_route_spec=spec,
            automation_defaults_sha256=digest(automation_defaults) if automation_defaults else None,
            status='geometry_only_not_locomotion_passed'))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2))
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--settings',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--automation-defaults',type=Path)
    a=p.parse_args();generate(a.inventory,a.settings,a.output,a.automation_defaults)
