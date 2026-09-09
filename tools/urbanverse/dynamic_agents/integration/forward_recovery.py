"""Explicit, logged forward restart after dynamic-contact falls or stalls."""
from __future__ import annotations
import math
import numpy as np


def dynamic_circles(vehicles,people,micro,catalog,specs):
    rows=[(np.asarray(a['position']),float(a['radius'])) for a in vehicles if a['status'] in ('moving','stopped')]
    rows.extend((np.asarray(p[:2]),.3) for p in people)
    lookup={s.agent_id:catalog[s.asset_id] for s in specs}
    rows.extend((s.position_xy,math.hypot(lookup[k].length_m,lookup[k].width_m)/2) for k,s in micro.items())
    return rows


class ForwardRecovery:
    def __init__(self,points,arc,ground_z,grid,free,settings):
        self.points=points;self.arc=arc;self.ground_z=ground_z;self.grid=grid;self.free=free;self.settings=settings
        self.events=[];self.segment=0;self.last_overlap=-math.inf;self.last_contact=-math.inf;self.waiting_since=None

    def observe(self,timestamp,position,nonfoot,circles):
        overlap=any(np.linalg.norm(position[:2]-p)<r+.5 for p,r in circles)
        if overlap:self.last_overlap=timestamp
        if overlap and nonfoot:self.last_contact=timestamp
        return overlap

    def proposal(self,arc_m,circles):
        offsets=np.arange(self.settings.get('forward_skip_m',1.),self.settings.get('maximum_safe_search_m',2.)+.01,.1)
        angles=np.linspace(0,2*np.pi,32,endpoint=False)
        disk=np.vstack([np.zeros((1,2)),*[r*np.c_[np.cos(angles),np.sin(angles)] for r in [.2,.4,.6]]])
        for offset in offsets:
            s=arc_m+offset
            if s>=self.arc[-1]-.1:continue
            xy=np.array([np.interp(s,self.arc,self.points[:,k]) for k in range(2)])
            if not self.grid.contains(self.free,xy+disk).all():continue
            if any(np.linalg.norm(xy-p)<r+.65 for p,r in circles):continue
            i=min(len(self.points)-2,int(np.searchsorted(self.arc,s)))
            direction=self.points[i+1]-self.points[i]
            return dict(arc_m=float(s),xy=xy.tolist(),yaw=float(math.atan2(direction[1],direction[0])),skipped_arc_m=float(offset))
        return None

    def apply(self,robot,external,controller,sim,action_manager,proposal,timestamp,position,reason):
        import torch
        if len(self.events)>=self.settings.get('maximum_recoveries',8):raise RuntimeError('Forward recovery budget exhausted')
        root=robot.data.default_root_state.clone();root[:,:]=0.
        root[0,:3]=torch.tensor([*proposal['xy'],self.ground_z+.4],device=root.device)
        yaw=proposal['yaw'];root[0,3]=math.cos(yaw/2);root[0,6]=math.sin(yaw/2)
        robot.reset()
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.joint_vel))
        action_manager.reset()
        if external is not None:external.reset()
        sim.forward()
        controller.resume_after_recovery(proposal['arc_m'],timestamp,self.settings.get('settle_s',1.))
        self.segment+=1;self.waiting_since=None
        event=dict(timestamp_s=timestamp,reason=reason,from_xyz=np.asarray(position).tolist(),
            to_xyz=[*proposal['xy'],self.ground_z+.4],segment_id=self.segment,
            physical_contact_recent=timestamp-self.last_contact<=5.,
            contact_attribution='non-foot force plus concurrent dynamic-body proximity; counterpart not identified by sensor',
            counts_as_walked_distance=False,**proposal)
        self.events.append(event);return event
