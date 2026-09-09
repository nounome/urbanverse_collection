"""Deterministic staggered residents on versioned closed paths, any scene."""
from __future__ import annotations
import numpy as np
from ..pedestrians.roaming import RoamingAssignment


def reduced_allocation(per_route, count):
    """Allow a lower configured population while retaining each route's capacity."""
    if per_route is None:
        return None
    raw=np.asarray(per_route)
    if raw.ndim!=1 or not np.isfinite(raw).all() or (raw<0).any() or not np.equal(raw,np.floor(raw)).all():
        raise ValueError('Route capacities must be nonnegative integers')
    capacity=raw.astype(int)
    if count<0 or count>capacity.sum():
        raise ValueError('Requested population exceeds admitted route capacity')
    if count==capacity.sum():
        return per_route
    allocation=np.zeros(len(capacity),dtype=int)
    for _ in range(count):
        scores=np.where(allocation<capacity,capacity/(allocation+1),-1.)
        allocation[int(np.argmax(scores))]+=1
    return allocation.tolist()


def distribute(rows,count,prefix,regions,reverse_alternate=False,per_route=None,
               minimum_start_separation_m=0.,excluded_start_xy=()):
    if not rows:raise ValueError('No admitted loops for '+prefix)
    lengths=np.array([r['length_m'] for r in rows]);allocation=np.zeros(len(rows),int)
    if per_route is None:
        for _ in range(count):
            i=int(np.argmax(lengths/(allocation+1)));allocation[i]+=1
    else:
        allocation=np.asarray(per_route)
        if allocation.shape!=lengths.shape or not np.isfinite(allocation).all() or (allocation<0).any() or not (allocation==allocation.astype(int)).all() or allocation.sum()!=count:
            raise ValueError('Per-route allocation must be nonnegative integers summing to count')
        allocation=allocation.astype(int)
    result={};occupied=[np.asarray(point,dtype=float) for point in excluded_start_xy]
    if any(point.shape != (2,) for point in occupied):
        raise ValueError('excluded_start_xy entries must be xy points')
    for i,(row,n) in enumerate(zip(rows,allocation)):
        points=np.asarray(row['xy'],float)[:-1]
        for rank in range(n):
            preferred=int((rank+.5)/n*len(points))%len(points)
            phase=preferred
            if occupied and minimum_start_separation_m>0:
                other=np.asarray(occupied)
                clearance=np.linalg.norm(points[:,None,:]-other[None,:,:],axis=2).min(axis=1)
                valid=np.flatnonzero(clearance>=minimum_start_separation_m)
                if not len(valid):
                    raise ValueError(
                        f'{prefix} closed-loop population cannot satisfy '
                        f'{minimum_start_separation_m:.2f} m global spawn spacing'
                    )
                cyclic=np.minimum((valid-preferred)%len(points),(preferred-valid)%len(points))
                phase=int(valid[int(np.argmin(cyclic))])
            path=np.roll(points,-phase,axis=0)
            if reverse_alternate and rank%2:path=np.vstack([path[0],path[:0:-1]])
            path=np.vstack([path,path[0]])
            component=regions.component_at(path[0])
            if not component or not regions.contains_points(path,component):raise ValueError('Loop crosses excluded component/domain')
            name=f'{prefix}_{len(result):02d}'
            result[name]=dict(xy=path,component_id=component,source_route=row['route_name'])
            occupied.append(path[0])
    return result


def people_assignments(loops,assets,ground_z):
    return tuple(RoamingAssignment(name,assets[i%len(assets)],item['component_id'],
        (*map(float,item['xy'][0]),ground_z),(*map(float,item['xy'][len(item['xy'])//2]),ground_z))
        for i,(name,item) in enumerate(loops.items()))
