"""Existing reservation-style yielding extended to narrow out/return loops."""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial import cKDTree
from ..vehicles.traffic_geometry import footprint,sat_intersects


class LoopPassage:
    def __init__(self,route,length,width,green_s=25.):
        self.length=route.length_m;self.ranges=[];self.owner=0;self.started=0.;self.closing=False;self.switches=0
        self.green_s=green_s;self.approach=8.
        far=int(np.argmax(np.linalg.norm(route.xy-route.xy[0],axis=1)))
        split=float(route.arc_m[far]);hits=[[],[]]
        for a,b in cKDTree(route.xy).query_pairs(math.hypot(length,width)+.7):
            sa,sb=route.arc_m[a],route.arc_m[b]
            if not sa<split<sb:continue
            separation=min(abs(sb-sa),self.length-abs(sb-sa))
            if separation<length*3:continue
            if math.cos(route.yaw[a]-route.yaw[b])>.7:continue
            if sat_intersects(footprint(route.xy[a],route.yaw[a],length,width),
                footprint(route.xy[b],route.yaw[b],length,width),margin=.5):
                hits[0].append(sa);hits[1].append(sb)
        if all(hits):self.ranges=[(max(0.,min(v)-length),min(self.length,max(v)+length)) for v in hits]

    def inside(self,arc):
        arc=arc%self.length
        return any(lo<=arc<=hi for lo,hi in self.ranges)

    def apply(self,agents,caps,timestamp):
        if not self.ranges:return
        occupants=[[],[]];waiting=[[],[]]
        for group,(lo,hi) in enumerate(self.ranges):
            for a in agents:
                s=a['route_arc_m']%self.length
                if lo<=s<=hi:occupants[group].append(a)
                elif 0<(lo-s)%self.length<=self.approach:waiting[group].append(a)
        other=1-self.owner
        if timestamp-self.started>=self.green_s and waiting[other]:self.closing=True
        if not occupants[self.owner] and waiting[other] and (self.closing or not waiting[self.owner]):
            self.owner=other;self.started=timestamp;self.closing=False;self.switches+=1
        for group in range(2):
            lo,_=self.ranges[group]
            admitted=None
            if group==self.owner and not self.closing and not occupants[group] and waiting[group]:
                # The short turnaround cannot store a whole convoy behind
                # the opposing queue. Admit one vehicle until it exits, then
                # allow the existing phase switch to drain the other side.
                admitted=min(waiting[group],key=lambda a:(lo-a['route_arc_m'])%self.length)['id']
            for a in waiting[group]:
                if a['id']==admitted:continue
                distance=(lo-a['route_arc_m'])%self.length
                cap=max(0.,distance-1.5)*.5
                caps[a['id']]=min(caps[a['id']],cap)

    def summary(self):
        return dict(kind='shared narrow out/return passage reservation',ranges_arc_m=self.ranges,
            owner=self.owner,switches=self.switches,green_s=self.green_s,
            maximum_transit_vehicles=1)
