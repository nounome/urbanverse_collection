"""Read-only conservative dynamic overlap labels; never a contact solver."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_rotation(yaw):
    return rotation([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])


def box(center, size, basis):
    c, s, r = np.asarray(center, float), np.asarray(size, float), np.asarray(basis, float)
    if c.shape != (3,) or s.shape != (3,) or r.shape != (3, 3):
        raise ValueError('Invalid box dimensions')
    if not all(np.isfinite(a).all() for a in (c, s, r)) or np.any(s <= 0):
        raise ValueError('Non-finite or empty box')
    return c, s/2, r


def intersects(a, b):
    """Full 15-axis OBB SAT, including vertical separation and tilted Go2."""
    ac, ah, ar = a; bc, bh, br = b
    axes = [*ar.T, *br.T, *(np.cross(x, y) for x in ar.T for y in br.T)]
    for axis in axes:
        if np.linalg.norm(axis) < 1e-8:
            continue
        if abs((bc-ac) @ axis) > np.abs(ar.T @ axis) @ ah + np.abs(br.T @ axis) @ bh + 1e-8:
            return False
    return True


def contains(b, point):
    c, h, r = b
    return bool(np.all(np.abs(r.T @ (np.asarray(point)-c)) <= h+1e-8))


class DynamicOverlapLabeler:
    @classmethod
    def from_configs(cls, scene_path, mixed_path, specs):
        from ..config import TrafficSceneConfig
        from ..micromobility.calibration import load_catalog
        scene = TrafficSceneConfig.load(Path(scene_path))
        mixed_path = Path(mixed_path).resolve()
        mixed = json.loads(mixed_path.read_text())
        catalog = load_catalog((mixed_path.parent/mixed['micromobility']['catalog']).resolve())
        dimensions = {s['agent_id']: dict(asset_id=s['asset_id'], size=[
            catalog[s['asset_id']].length_m, catalog[s['asset_id']].width_m,
            catalog[s['asset_id']].height_m]) for s in specs}
        return cls(json.loads(scene.vehicle_catalog.read_text())['records'], dimensions,
                   person_radius=mixed['people'].get('radius_m', .3),
                   micro_ground_z=mixed['micromobility'].get('visual_support_z_m'))

    def __init__(self, vehicle_records, micro_dimensions, *, person_radius=.3,
                 person_height=1.7, micro_ground_z=None, go2_size=(.75, .5, .6)):
        self.vehicles = {r['asset_id']: r for r in vehicle_records}
        self.micro = micro_dimensions
        self.person_radius = person_radius
        self.person_height = person_height
        self.micro_ground_z = micro_ground_z
        self.go2_size = go2_size

    def evaluate(self, row, camera_position=None, camera_pose_source='not_available'):
        quat = row.get('base_quaternion_wxyz')
        basis = rotation(quat) if quat is not None else yaw_rotation(row['base_yaw_rad_world'])
        robot = box(row['base_position_world'], self.go2_size, basis)
        actors = []
        for a in (row.get('scene10_multivehicle_traffic') or {}).get('agents', []):
            if not a.get('active_on_road'):
                continue
            record = self.vehicles[a['asset_id']]
            height = 2*record['grounding']['body_center_height_above_support_m']
            size = [*record['footprint']['length_width_m'], height]
            c = [*a['center_xyz'][:2], a['ground_z']+height/2]
            actors.append((f"V{int(a['id']):02d}", 'vehicle', a['asset_id'], box(c, size, yaw_rotation(math.radians(a['heading_deg'])))))
        ids = row.get('pedestrian_ids', [])
        for i, xyz in enumerate(row.get('pedestrian_positions_xyz', [])):
            name = str(ids[i]) if i < len(ids) else f'P{i:02d}'
            c = [*xyz[:2], xyz[2]+self.person_height/2]
            actors.append((name, 'pedestrian', None, box(c, [2*self.person_radius]*2+[self.person_height], np.eye(3))))
        for name, a in row.get('micromobility_states', {}).items():
            dim = self.micro[name]
            z = a.get('ground_z', self.micro_ground_z)
            if z is None:
                raise ValueError('Micromobility support height unavailable')
            c = [*a['position_xy'], z+dim['size'][2]/2]
            actors.append((name, 'micromobility', dim['asset_id'], box(c, dim['size'], yaw_rotation(a['yaw_rad']))))
        hits = []
        for name, kind, asset, bounds in actors:
            body_hit = intersects(robot, bounds)
            camera_hit = contains(bounds, camera_position) if camera_position is not None else None
            if body_hit or camera_hit:
                hits.append(dict(actor_id=name, category=kind, asset_id=asset,
                                 body_overlap_candidate=bool(body_hit), camera_inside_candidate=camera_hit))
        return dict(timestamp_s=float(row['timestamp_s']), step_index=row.get('step_index'),
                    motion_segment_id=row.get('motion_segment_id', 0),
                    method='conservative_3d_obb_v1_not_mesh_or_contact',
                    go2_orientation_source='full_quaternion' if quat is not None else 'legacy_yaw_only',
                    camera_pose_source=camera_pose_source, camera_evaluated=camera_position is not None,
                    camera_position_world=None if camera_position is None else list(map(float, camera_position)),
                    actors=hits, evaluated_actor_count=len(actors),
                    nonfoot_contact_observed=bool(row.get('contact_classification', {}).get('non_foot_collision', False)),
                    contact_actor_attribution='unavailable',
                    exclude_from_ordinary_navigation_candidate=bool(hits))


class OverlapRecorder:
    """Sidecars preserve samples, exact capture timestamps and bounded intervals."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.samples = (self.directory/'overlap_samples.jsonl').open('x', buffering=1)
        self.frames = (self.directory/'overlap_frames.jsonl').open('x', buffering=1)
        self.active = {}; self.events = []; self.last_time = None; self.last_segment = None
        self.count = 0; self.frame_count = 0; self.closed = False

    def observe(self, record):
        t = record['timestamp_s']; segment = record['motion_segment_id']
        if self.last_time is not None and t <= self.last_time:
            raise ValueError('Overlap sample timestamps must increase')
        if self.last_segment is not None and segment != self.last_segment:
            self._end_all(None, True)
        keys = {}
        for actor in record['actors']:
            for flag in ('body_overlap_candidate', 'camera_inside_candidate'):
                if actor[flag]: keys[(actor['actor_id'], flag)] = actor
        for key in list(self.active):
            if key not in keys:
                self.events.append(dict(self.active.pop(key), first_negative_time_s=t, right_censored=False))
        for key, actor in keys.items():
            if key not in self.active:
                self.active[key] = dict(actor_id=key[0], category=actor['category'], kind=key[1],
                    motion_segment_id=segment, first_positive_time_s=t,
                    preceding_negative_time_s=self.last_time if segment == self.last_segment else None,
                    left_censored=self.last_time is None or segment != self.last_segment)
            self.active[key]['last_positive_time_s'] = t
        self.samples.write(json.dumps(record)+'\n')
        self.last_time=t; self.last_segment=segment; self.count+=1

    def frame(self, record, frame_index, *, timestamp_s, camera_id, alignment='exact_runtime_capture'):
        self.frames.write(json.dumps(dict(record, frame_index=int(frame_index),
            capture_timestamp_s=float(timestamp_s), camera_id=camera_id,
            temporal_alignment=alignment,
            sample_offset_s=float(record['timestamp_s']-timestamp_s)))+'\n')
        self.frame_count+=1

    def _end_all(self, negative, censored):
        for item in self.active.values():
            self.events.append(dict(item, first_negative_time_s=negative, right_censored=censored))
        self.active.clear()

    def close(self, completed=False):
        if self.closed: return
        self._end_all(None, True)
        self.samples.close(); self.frames.close()
        payload = dict(schema_version=1, completed=completed, sample_count=self.count,
            frame_count=self.frame_count, events=self.events,
            scope='Dynamic actors only; conservative boxes, not exact meshes or physical contact. No static obstacle or general visual occlusion coverage.',
            interval_semantics='First/last positive samples plus surrounding negative brackets; EOF and segment boundaries remain censored. No interpolation across reset.')
        (self.directory/'overlap_events.json').write_text(json.dumps(payload, indent=2))
        self.closed=True
