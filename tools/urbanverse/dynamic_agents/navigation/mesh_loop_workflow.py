"""Portable mesh-map and closed-loop candidate builder (no simulation side effects)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, distance_transform_edt

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from urbanverse.dynamic_agents.traffic.routes import resample_polyline
from urbanverse.dynamic_agents.navigation.global_route_planner import astar
from urbanverse.dynamic_agents.navigation.automation_rules import loop_rank, allocate_capacity, shared_capacity
from urbanverse.dynamic_agents.navigation.cleanup_categories import cleanup_category


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MeshMap:
    def __init__(self, inventory:Path):
        self.path=inventory.resolve();self.meta=json.loads(self.path.read_text())
        if not self.meta['flat_ground_gate']:raise ValueError('Ground not qualified for flat map')
        self.view=np.array(self.meta['view_xy']);self.w,self.h=self.meta['resolution']
        self.pixel=(self.view[2:]-self.view[:2])/np.array([self.w,self.h])
        self.obstacles=self.meta['obstacles']
        self.ground={n:np.asarray(Image.open(self.path.parent/f'ground_{n}.png'))>0 for n in ('Lane','Sidewalk','NearRoad','NearBuffer')}
        # Matches the established semantic-map draw precedence, but now at
        # native navigation resolution instead of rasterizing a review image.
        self.semantic=np.zeros((self.h,self.w),np.uint8)
        for i,n in enumerate(self.ground,1):self.semantic[self.ground[n]]=i
        self.lane=self.semantic==1
        self.shared=np.isin(self.semantic,[3,4])

    def pixels(self,xy):
        p=np.asarray(xy);return np.stack([(p[...,0]-self.view[0])/self.pixel[0],(self.view[3]-p[...,1])/self.pixel[1]],axis=-1)

    def world(self,pixels):
        p=np.asarray(pixels);return np.stack([self.view[0]+(p[...,0]+.5)*self.pixel[0],self.view[3]-(p[...,1]+.5)*self.pixel[1]],axis=-1)

    def contains(self,mask,xy):
        p=np.floor(self.pixels(xy)).astype(int)
        valid=(p[...,0]>=0)&(p[...,0]<self.w)&(p[...,1]>=0)&(p[...,1]<self.h)
        return valid&mask[np.clip(p[...,1],0,self.h-1),np.clip(p[...,0],0,self.w-1)]

    def occupancy(self,removed=()):
        mask=np.zeros((self.h,self.w),np.uint8)
        for row in self.obstacles:
            if row['instance_root'] in removed:continue
            for poly in row['polygons_xy']:
                pix=np.rint(self.pixels(poly)-.5).astype(np.int32)
                cv2.fillPoly(mask,[pix],255)
        return mask>0


def open_vehicle_routes(grid, free, profile):
    """Find long straight streams with dense full-body clearance, no U-turn."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    candidates = []
    for component in sorted(range(1, count), key=lambda c: stats[c, 4], reverse=True)[:12]:
        cells = np.argwhere(labels == component)
        if len(cells) < 20:
            continue
        world = grid.world(cells[:, ::-1])
        centre = world.mean(axis=0)
        _, axes = np.linalg.eigh(np.cov(world.T))
        angles = [0., np.pi / 2, float(np.arctan2(axes[1, -1], axes[0, -1]))]
        for angle in angles:
            axis = np.array([np.cos(angle), np.sin(angle)])
            side = np.array([-axis[1], axis[0]])
            longitudinal = (world-centre) @ axis
            lateral = (world-centre) @ side
            arcs = np.arange(longitudinal.min(), longitudinal.max(), .1)
            for offset in np.arange(lateral.min(), lateral.max()+.01, .25):
                xy = centre + arcs[:, None]*axis + offset*side
                inside = grid.contains(free, xy)
                # Cheap centre gate first; rectangles are checked for every accepted sample.
                edges = np.diff(np.r_[False, inside, False].astype(int))
                for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if (end-begin)*.1 < 8.:
                        continue
                    segment = xy[begin:end]
                    good = body_clear(grid, free, segment, np.full(len(segment), angle),
                                      profile['length_m'], profile['width_m'], profile.get('margin_m', .2))
                    boundaries = np.diff(np.r_[False, good, False].astype(int))
                    for a, b in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
                        length = (b-a-1)*.1
                        if length < 8.:
                            continue
                        candidates.append(dict(xy=segment[a:b].tolist(), yaw_deg=[float(np.degrees(angle))]*(b-a),
                            length_m=length, minimum_turning_radius_m=profile['minimum_radius_m'],
                            max_curvature_radpm=0., closed=False, component_id=component,
                            admission='dense whole-body mesh clearance; open spawn/despawn stream'))
    return sorted(candidates, key=lambda row: row['length_m'], reverse=True)[:1]


def periodic_route(points,spacing=.25,sigma_m=1.):
    p=np.asarray(points,float)
    if np.linalg.norm(p[0]-p[-1])>1e-8:p=np.vstack([p,p[0]])
    p=resample_polyline(p,spacing)[:-1]
    p=gaussian_filter1d(p,sigma_m/spacing,axis=0,mode='wrap')
    p=resample_polyline(np.vstack([p,p[0]]),spacing)[:-1]
    delta=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
    yaw=np.arctan2(delta[:,1],delta[:,0])
    ds=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    dyaw=(np.roll(yaw,-1)-yaw+np.pi)%(2*np.pi)-np.pi
    curvature=np.abs(dyaw)/np.maximum(ds,1e-6)
    return p,yaw,float(ds.sum()),float(curvature.max())


def body_clear(grid:MeshMap,free,points,yaws,length,width,margin=.15):
    # Whole rectangular footprint, including interiors; no four-corner-only gate.
    spacing=min(.1,float(grid.pixel.min()))
    xs=np.linspace(-length/2-margin,length/2+margin,max(3,int(np.ceil((length+2*margin)/spacing))+1))
    ys=np.linspace(-width/2-margin,width/2+margin,max(3,int(np.ceil((width+2*margin)/spacing))+1))
    good=np.ones(len(points),bool)
    x,y=np.meshgrid(xs,ys);x=x.ravel()[None,:];y=y.ravel()[None,:]
    for start in range(0,len(points),128):
        stop=min(start+128,len(points))
        co,si=np.cos(yaws[start:stop,None]),np.sin(yaws[start:stop,None])
        q=points[start:stop,None,:]+np.stack([co*x-si*y,si*x+co*y],axis=-1)
        good[start:stop]=grid.contains(free,q).all(axis=1)
    return good


def connect_turning_domains(opened,distance,inset,grid,profile):
    """Join turn-capable bulbs by a two-way corridor, not narrow dead ends.

    A* finds a centre spine; its two contour sides are the out/return paths.
    This is candidate construction only: full body and curvature gates follow.
    """
    count,labels,stats,centres=cv2.connectedComponentsWithStats(opened,connectivity=8)
    components=[i for i in range(1,count) if stats[i,4]*np.prod(grid.pixel)>10.]
    if len(components)<2:return opened
    half_width=profile['width_m']*.75+profile.get('margin_m',.15)
    step=3
    coarse=distance[::step,::step]
    valid=coarse>=inset+half_width+.2
    anchors={}
    for i in components:
        yy,xx=np.where((labels[::step,::step]==i)&valid)
        if len(xx):
            k=int(np.argmax(coarse[yy,xx]));anchors[i]=(int(yy[k]),int(xx[k]))
    pairs=sorted(((float(np.linalg.norm(centres[a]-centres[b])),a,b)
        for a in anchors for b in anchors if a<b),reverse=True)
    result=opened.copy()
    for _,a,b in pairs[:6]:
        path=astar(anchors[a],anchors[b],valid,coarse,3.,float(grid.pixel.mean()*step),6.,1.)
        if path is None:continue
        points=np.array([(x*step,y*step) for y,x in path],np.int32)
        cv2.polylines(result,[points],False,1,int(2*half_width/grid.pixel.mean()))
    result[distance<inset]=0
    return result


def loop_candidates(grid,free,profile,max_components=12):
    distance=distance_transform_edt(free,sampling=grid.pixel[::-1])
    best={};attempts=[]
    radius=profile['minimum_radius_m']
    for inset in profile['insets_m']:
        safe=(distance>=inset).astype(np.uint8)
        # Remove narrow dead-end fingers from the *candidate centre domain*.
        # The original free mask remains unchanged for all admission checks.
        # Otherwise Gaussian smoothing of a long thin finger still produces an
        # impossible U-turn, however many smoothing strengths are attempted.
        opening=profile.get('turning_opening_m', radius*1.25)
        if opening>0:
            rx,ry=np.ceil(opening/grid.pixel).astype(int)
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rx+1,2*ry+1))
            safe=cv2.morphologyEx(safe,cv2.MORPH_OPEN,kernel)
            if profile.get('connect_turning_domains',False):
                safe=connect_turning_domains(safe,distance,inset,grid,profile)
        count,labels,stats,_=cv2.connectedComponentsWithStats(safe,connectivity=8)
        ranked=sorted(range(1,count),key=lambda x:stats[x,cv2.CC_STAT_AREA],reverse=True)[:max_components]
        for component in ranked:
            if stats[component,cv2.CC_STAT_AREA]*np.prod(grid.pixel)<profile.get('minimum_area_m2',10):continue
            contours,_=cv2.findContours((labels==component).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:continue
            contour=max(contours,key=cv2.contourArea).reshape(-1,2)
            if len(contour)<4:continue
            raw=grid.world(contour)
            for sigma in profile['smoothing_m']:
                xy,yaw,length,kappa=periodic_route(raw,sigma_m=sigma)
                if length<profile.get('minimum_length_m',15):continue
                if radius>0 and kappa>1/radius+1e-5:
                    attempts.append(dict(length_m=length,reason='curvature',max_curvature=kappa));continue
                clear=body_clear(grid,free,xy,yaw,profile['length_m'],profile['width_m'],profile.get('margin_m',.15))
                if not clear.all():
                    attempts.append(dict(length_m=length,reason='footprint',bad_samples=int((~clear).sum())));continue
                # Deduplicate nested candidates belonging to the same original component.
                key=int(profile['_original_labels'][int(grid.pixels(xy[0])[1]),int(grid.pixels(xy[0])[0])])
                row=dict(component_id=key,xy=np.vstack([xy,xy[0]]).tolist(),
                    yaw_deg=np.degrees(np.r_[np.unwrap(yaw),np.unwrap(yaw)[-1]+((yaw[0]-yaw[-1]+np.pi)%(2*np.pi)-np.pi)]).tolist(),
                    length_m=length,max_curvature_radpm=kappa,inset_m=inset,smoothing_m=sigma,
                    minimum_turning_radius_m=radius,closed=True,admission='dense whole-body raster and curvature passed')
                rank = lambda r: loop_rank(r, profile.get('coverage_cell_m', 1.0)) if profile.get('rank_by_coverage') else r['length_m']
                if key not in best or rank(row)>rank(best[key]):best[key]=row
    return sorted(best.values(),key=rank,reverse=True) if best else [],attempts


def bulb_loop(grid,free,profile):
    """Search between far-apart turn spaces using real forward-only curves."""
    from urbanverse.dynamic_agents.navigation.curved_loop_search import search
    distance=distance_transform_edt(free,sampling=grid.pixel[::-1])
    radius=profile['minimum_radius_m']
    threshold=profile['minimum_radius_m']+profile['width_m']/2+profile.get('margin_m',.15)
    count,labels,stats,_=cv2.connectedComponentsWithStats((distance>=threshold).astype(np.uint8))
    centres=[]
    for i in range(1,count):
        if stats[i,4]*np.prod(grid.pixel)<1.:continue
        yy,xx=np.where(labels==i);k=int(np.argmax(distance[yy,xx]));centres.append(grid.world([xx[k],yy[k]]))
    pairs=sorted(((float(np.linalg.norm(a-b)),a,b) for i,a in enumerate(centres) for b in centres[i+1:]),key=lambda p:p[0],reverse=True)
    enclosing=np.hypot(profile['length_m']/2+.2,profile['width_m']/2+.2)+.15
    def clear(xy,yaw):
        pixels=np.floor(grid.pixels(xy)).astype(int)
        d=distance[np.clip(pixels[:,1],0,grid.h-1),np.clip(pixels[:,0],0,grid.w-1)]
        ok=grid.contains(free,xy);check=ok&(d<enclosing)
        if check.any():ok[check]=body_clear(grid,free,xy[check],yaw[check],profile['length_m'],profile['width_m'],profile.get('margin_m',.15))
        return ok
    logs=[]
    for span,a,b in pairs[:2]:
        if span<30:continue
        axis=(a-b)/span;side=np.array([axis[1],-axis[0]])
        start=a+axis*radius;goal=b-axis*radius
        yaw0=float(np.arctan2(side[1],side[0]));yaw1=yaw0+np.pi
        if not clear(np.array([start,goal]),np.array([yaw0,yaw1])).all():continue
        legs=[]
        for p,y,q,v in [(start,yaw0,goal,yaw1),(goal,yaw1,start,yaw0)]:
            print('hybrid loop leg',np.round(p,2).tolist(),'->',np.round(q,2).tolist(),flush=True)
            leg,info=search(p,y,q,v,clear,radius,maximum_expansions=profile.get('maximum_hybrid_expansions',25000))
            logs.append(info);print('hybrid loop result',info,flush=True)
            if leg is None:break
            legs.append(leg)
        if len(legs)==2:
            xy=np.vstack([legs[0][0],legs[1][0][1:]])
            yaw=np.unwrap(np.r_[legs[0][1],legs[1][1][1:]])
            length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
            return [dict(component_id=1,xy=xy.tolist(),yaw_deg=np.degrees(yaw).tolist(),length_m=length,
                minimum_turning_radius_m=profile['minimum_radius_m'],closed=True,route_name='vehicles_bulb_loop',
                max_curvature_radpm=1/radius,admission='whole-body hybrid A* with exact CSC closure',search=logs)],logs
    return [],logs


def capsule_loops(grid,free,profile):
    """Exact circular end turns for long, narrow straight components."""
    distance=distance_transform_edt(free,sampling=grid.pixel[::-1]);result=[]
    for radius in [profile['minimum_radius_m']+.15,profile['minimum_radius_m']+.2,profile['minimum_radius_m']+.25]:
        margin=profile.get('margin_m',.15)
        extent=np.hypot(radius+profile['width_m']/2+margin,profile['length_m']/2+margin)+.025
        n,labels,stats,_=cv2.connectedComponentsWithStats((distance>=extent).astype(np.uint8))
        for i in range(1,n):
            if stats[i,4]<20:continue
            yy,xx=np.where(labels==i);points=grid.world(np.c_[xx,yy])
            _,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1]
            projection=points@axis;order=np.argsort(projection)
            a=points[order[:max(1,len(points)//100)]].mean(0)
            b=points[order[-max(1,len(points)//100):]].mean(0)
            span=np.linalg.norm(b-a)
            if span<10.:continue
            axis=(b-a)/span;side=np.array([axis[1],-axis[0]])
            line0=np.linspace(a+radius*side,b+radius*side,max(2,int(span/.15)+1))
            angles=np.linspace(0,np.pi,int(np.pi*radius/.15)+1)
            arc0=b+radius*(np.cos(angles)[:,None]*side+np.sin(angles)[:,None]*axis)
            line1=np.linspace(b-radius*side,a-radius*side,len(line0))
            arc1=a-radius*(np.cos(angles)[:,None]*side+np.sin(angles)[:,None]*axis)
            p=np.vstack([line0,arc0[1:],line1[1:],arc1[1:]])[:-1]
            delta=np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
            yaw=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
            if not body_clear(grid,free,p,yaw,profile['length_m'],profile['width_m'],margin).all():continue
            length=float(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum())
            result.append(dict(component_id=i,xy=np.vstack([p,p[0]]).tolist(),
                yaw_deg=np.degrees(np.r_[yaw,yaw[-1]+((yaw[0]-yaw[-1]+np.pi)%(2*np.pi)-np.pi)]).tolist(),
                length_m=length,minimum_turning_radius_m=profile['minimum_radius_m'],closed=True,
                max_curvature_radpm=1/radius,admission='exact capsule + dense whole-body raster passed'))
    return result


def circle_loops(grid,free,profile):
    """Build conservative loops inside large open regions.

    The clearance proof uses the largest empty circle in each connected free
    component.  Keeping the route circle plus the complete body bounding radius
    inside that empty circle makes this useful for plazas without weakening the
    existing footprint or curvature checks.
    """
    distance=distance_transform_edt(free,sampling=grid.pixel[::-1]);result=[]
    count,labels,stats,_=cv2.connectedComponentsWithStats(free.astype(np.uint8),connectivity=8)
    body_radius=np.hypot(profile['length_m']/2+profile.get('margin_m',.15),
                         profile['width_m']/2+profile.get('margin_m',.15))
    for component in range(1,count):
        if stats[component,cv2.CC_STAT_AREA]*np.prod(grid.pixel)<profile.get('minimum_area_m2',10):continue
        yy,xx=np.where(labels==component)
        if not len(xx):continue
        index=int(np.argmax(distance[yy,xx]));clearance=float(distance[yy[index],xx[index]])
        radius=clearance-body_radius-.1
        if radius<profile['minimum_radius_m']+.05:continue
        centre=grid.world([xx[index],yy[index]])
        samples=max(24,int(np.ceil(2*np.pi*radius/.15)))
        angles=np.linspace(0,2*np.pi,samples,endpoint=False)
        p=centre+radius*np.c_[np.cos(angles),np.sin(angles)]
        yaw=np.unwrap(angles+np.pi/2)
        margin=profile.get('margin_m',.15)
        if not body_clear(grid,free,p,yaw,profile['length_m'],profile['width_m'],margin).all():continue
        length=float(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum())
        result.append(dict(component_id=component,xy=np.vstack([p,p[0]]).tolist(),
            yaw_deg=np.degrees(np.r_[yaw,yaw[-1]+2*np.pi/samples]).tolist(),length_m=length,
            minimum_turning_radius_m=profile['minimum_radius_m'],closed=True,
            max_curvature_radpm=1/radius,
            admission='largest-empty-circle + dense whole-body raster passed'))
    return result


def build(inventory,settings_path,out,only_group=None,automation_defaults=None):
    out.mkdir(parents=True,exist_ok=False)
    settings=json.loads(settings_path.read_text());grid=MeshMap(inventory)
    if automation_defaults is not None:
        defaults=json.loads(Path(automation_defaults).read_text())
        # Scene overrides win, including a deliberately stricter cleanup cap.
        settings.setdefault('maximum_cleanup_objects',defaults['maximum_cleanup_objects'])
        settings.setdefault('population',defaults['population'])
        for profile in settings['profiles'].values():
            profile.setdefault('rank_by_coverage',defaults['rank_by_coverage'])
            profile.setdefault('coverage_cell_m',defaults['coverage_cell_m'])
    removed=settings.get('removed_instance_roots',[])
    for root in removed:
        if cleanup_category(root) is None:
            raise ValueError('Removal requires explicit small-obstacle category: '+root)
        if not any(r['instance_root']==root for r in grid.obstacles):raise ValueError('Unknown removed root '+root)
    cleanup_limit=settings.get('maximum_cleanup_objects',6)
    # Explicit null enables the newly authorized uncapped quantity. Categories
    # and exact known roots are still validated; this is not an auto remover.
    if cleanup_limit is not None and len(removed)>cleanup_limit:raise ValueError('Cleanup limit exceeded')
    occupied=grid.occupancy(removed)
    before=grid.occupancy()
    result=dict(schema_version=1,map_inventory=str(inventory.resolve()),settings=str(settings_path.resolve()),
        input_sha256=digest(inventory),settings_sha256=digest(settings_path),removed_instance_roots=removed,
        geometry_only=True,runtime_verified=False,groups={})
    result['effective_settings']=settings
    result['automation_defaults_sha256']=digest(automation_defaults) if automation_defaults else None
    for name,profile in settings['profiles'].items():
        if only_group is not None and name!=only_group:continue
        domain=grid.lane if name=='vehicles' else grid.shared
        free=domain&~occupied
        if name=='vehicles' and settings.get('vehicle_mode')=='not_applicable_no_lane':
            if grid.lane.any():raise ValueError('No-Lane exemption requires an empty Lane mask')
            result['groups'][name]=dict(routes=[],candidate_count=0,attempt_count=0,
                rejection_examples=[],profile=profile,status='not_applicable_no_lane')
            Image.fromarray((free*255).astype(np.uint8)).save(out/f'{name}_free.png')
            continue
        rank = lambda r: loop_rank(r, profile.get('coverage_cell_m', 1.0)) if profile.get('rank_by_coverage') else r['length_m']
        _,original_labels=cv2.connectedComponents(free.astype(np.uint8),connectivity=8)
        candidates,attempts=loop_candidates(grid,free,{**profile,'_original_labels':original_labels})
        if name=='micromobility' or (name=='vehicles' and profile.get('capsule_out_and_back',False)):
            candidates+=capsule_loops(grid,free,profile)
            candidates+=circle_loops(grid,free,profile)
            unique={}
            for row in candidates:
                x,y=np.floor(grid.pixels(row['xy'][0])).astype(int);key=int(original_labels[y,x])
                if key not in unique or rank(row)>rank(unique[key]):unique[key]=row
            candidates=sorted(unique.values(),key=rank,reverse=True)
        if profile.get('hybrid_bulb_connection'):
            extra,search_log=bulb_loop(grid,free,profile)
            candidates=sorted(candidates+extra,key=rank,reverse=True)
            attempts.extend(search_log)
        selected=candidates[:1] if name=='vehicles' else candidates
        if name=='vehicles' and not selected and only_group is None and settings.get('vehicle_open_fallback'):
            selected=open_vehicle_routes(grid,free,profile)
        for i,row in enumerate(selected):row.update(route_name=f'{name}_loop_{i:02d}',desired_speed_mps=profile['speed_mps'])
        result['groups'][name]=dict(routes=selected,candidate_count=len(candidates),attempt_count=len(attempts),
            rejection_examples=attempts[:12],profile=profile,status='planned' if selected else 'no_feasible_loop')
        Image.fromarray((free*255).astype(np.uint8)).save(out/f'{name}_free.png')
        print(name,len(selected),[round(r['length_m'],1) for r in selected],flush=True)
    if settings.get('population') and only_group is None:
        pop=settings['population'];pixel_area=float(np.prod(grid.pixel))
        lengths={n:[r['length_m'] for r in g['routes']] for n,g in result['groups'].items()}
        result['population_proposal']={'vehicles':allocate_capacity(
            float((grid.lane&~occupied).sum())*pixel_area,lengths['vehicles'],**pop['vehicles'])}
        if result['groups']['vehicles']['routes'] and pop['vehicles']['maximum_count']>0 and result['population_proposal']['vehicles']['count']==0:
            # One car can traverse an admitted short stream without an inter-car gap.
            result['population_proposal']['vehicles'].update(count=1,per_route=[1],status='single_vehicle_short_route')
        # One shared area budget per connected component; a large remote
        # sidewalk must not grant capacity to a tiny disconnected island.
        shared_free=grid.shared&~occupied
        _,labels,stats,_=cv2.connectedComponentsWithStats(shared_free.astype(np.uint8),connectivity=8)
        names=('people','micromobility')
        components={n:[] for n in names}
        for n in names:
            for row in result['groups'][n]['routes']:
                x,y=np.floor(grid.pixels(row['xy'][0])).astype(int)
                components[n].append(int(labels[y,x]))
        allocations={n:[0]*len(lengths[n]) for n in names}
        component_records=[]
        for c in sorted(set(components['people']+components['micromobility'])):
            ids={n:[i for i,v in enumerate(components[n]) if v==c] for n in names}
            proposal=shared_capacity(float(stats[c,cv2.CC_STAT_AREA])*pixel_area,
                {n:[lengths[n][i] for i in ids[n]] for n in names},pop)
            for n in names:
                for i,value in zip(ids[n],proposal[n]['per_route']):allocations[n][i]=value
            component_records.append(dict(component_id=c,proposal=proposal))
        for n in names:
            while sum(allocations[n])>pop[n]['maximum_count']:
                i=max(range(len(allocations[n])),key=lambda j:allocations[n][j]/max(lengths[n][j],1e-9))
                allocations[n][i]-=1
            result['population_proposal'][n]=dict(count=sum(allocations[n]),per_route=allocations[n],
                component_ids=components[n],status='proposed_requires_joint_spawn_checks')
        result['population_components']=component_records
        if result['groups']['vehicles'].get('status')=='not_applicable_no_lane':
            result['population_proposal']['vehicles']['status']='not_applicable_no_lane'
    Image.fromarray((occupied*255).astype(np.uint8)).save(out/'obstacles.png')
    Image.fromarray((before*255).astype(np.uint8)).save(out/'obstacles_before_cleanup.png')
    colour=np.full((grid.h,grid.w,3),(42,48,54),np.uint8)
    for label,c in [(1,(72,173,104)),(2,(211,67,67)),(3,(66,129,201)),(4,(230,184,63))]:colour[grid.semantic==label]=c
    Image.fromarray(colour).save(out/'semantic_categories.png')
    (out/'planned_loops.json').write_text(json.dumps(result,indent=2))
    (out/'map_transform.json').write_text(json.dumps(dict(view_xy=grid.view.tolist(),resolution=[grid.w,grid.h],ground_z_m=grid.meta['ground_z_m']),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--only-group',choices=['vehicles','people','micromobility'])
    p.add_argument('--automation-defaults',type=Path,help='Opt into coverage ranking and capacity proposals; scene overrides win')
    a=p.parse_args();build(a.inventory,a.settings,a.output_dir,a.only_group,a.automation_defaults)
