"""Bounded forward-only search with exact circular/straight goal connections."""
from __future__ import annotations
import heapq
import math
import numpy as np


def csc_paths(start,yaw0,goal,yaw1,radius,spacing=.2):
    """Four Dubins CSC families, sampled including exact end poses."""
    start,goal=np.asarray(start),np.asarray(goal)
    result=[]
    for a in (-1,1):
        for b in (-1,1):
            c0=start+a*radius*np.array([-math.sin(yaw0),math.cos(yaw0)])
            c1=goal+b*radius*np.array([-math.sin(yaw1),math.cos(yaw1)])
            d=c1-c0;distance=np.linalg.norm(d)
            if distance<abs(a-b)*radius+1e-8:continue
            heading=math.atan2(d[1],d[0])+math.asin((a-b)*radius/distance)
            theta0=yaw0-a*math.pi/2;theta_mid0=heading-a*math.pi/2
            theta_mid1=heading-b*math.pi/2;theta1=yaw1-b*math.pi/2
            sweep0=(a*(theta_mid0-theta0))%(2*math.pi)
            sweep1=(b*(theta1-theta_mid1))%(2*math.pi)
            q0=c0+radius*np.array([math.cos(theta_mid0),math.sin(theta_mid0)])
            q1=c1+radius*np.array([math.cos(theta_mid1),math.sin(theta_mid1)])
            line=np.linalg.norm(q1-q0)
            angles0=theta0+a*np.linspace(0,sweep0,max(2,int(math.ceil(sweep0*radius/spacing))+1))
            angles1=theta_mid1+b*np.linspace(0,sweep1,max(2,int(math.ceil(sweep1*radius/spacing))+1))
            xy0=c0+radius*np.c_[np.cos(angles0),np.sin(angles0)]
            xy1=np.linspace(q0,q1,max(2,int(math.ceil(line/spacing))+1))
            xy2=c1+radius*np.c_[np.cos(angles1),np.sin(angles1)]
            xy=np.vstack([xy0,xy1[1:],xy2[1:]])
            yaw=np.r_[angles0+a*math.pi/2,np.full(len(xy1)-1,heading),angles1[1:]+b*math.pi/2]
            result.append((radius*(sweep0+sweep1)+line,xy,np.unwrap(yaw)))
    return sorted(result,key=lambda x:x[0])


def search(start,yaw0,goal,yaw1,clear,radius,maximum_expansions=45000):
    """Hybrid A*, metre frame; clear accepts batches of xy/yaw samples."""
    def key(p,y):return (int(round(p[0]/.5)),int(round(p[1]/.5)),int(round(y/(2*math.pi)*72))%72)
    initial=key(start,yaw0);pose={initial:(np.array(start,float),yaw0)};cost={initial:0.};parents={};segments={}
    queue=[(float(np.linalg.norm(np.asarray(start)-goal)),0.,initial)]
    terminal=None;shot=None;visited=0;closest=float(np.linalg.norm(np.asarray(start)-goal));closest_pose=np.asarray(start)
    for visited in range(maximum_expansions):
        if not queue:break
        _,g,k=heapq.heappop(queue)
        if g>cost.get(k,math.inf)+1e-8:continue
        p,y=pose[k];distance=np.linalg.norm(p-goal)
        if distance<closest:closest=float(distance);closest_pose=p
        if distance<22. and visited%5==0:
            for length,xy,yaw in csc_paths(p,y,goal,yaw1,radius):
                if length>distance+radius*2.5:continue
                if clear(xy,yaw).all():terminal=k;shot=(xy,yaw);break
            if terminal is not None:break
        for curvature in np.array([-1.,-.5,0,.5,1.])/radius:
            ds=np.arange(1,5)*.25
            headings=y+curvature*ds
            if abs(curvature)<1e-9:xy=p+ds[:,None]*np.array([math.cos(y),math.sin(y)])
            else:xy=p+np.c_[(np.sin(headings)-math.sin(y))/curvature,(-np.cos(headings)+math.cos(y))/curvature]
            if not clear(xy,headings).all():continue
            q=key(xy[-1],headings[-1]);ng=g+1.+.2*abs(curvature)
            if ng>=cost.get(q,math.inf)-1e-8:continue
            cost[q]=ng;pose[q]=(xy[-1],headings[-1]);parents[q]=k;segments[q]=(xy,headings)
            heapq.heappush(queue,(ng+float(np.linalg.norm(xy[-1]-goal)),ng,q))
    if terminal is None:return None,dict(expansions=visited+1,status='exhausted',closest_goal_distance_m=closest,closest_xy=closest_pose.tolist())
    legs=[shot];cursor=terminal
    while cursor!=initial:legs.append(segments[cursor]);cursor=parents[cursor]
    legs.reverse()
    xy=np.vstack([np.array(start)[None,:],*[v[0] for v in legs]])
    yaw=np.unwrap(np.concatenate([np.array([yaw0]),*[v[1] for v in legs]]))
    keep=np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-7]
    return (xy[keep],yaw[keep]),dict(expansions=visited+1,status='found')
