"""Three-camera writer for the mixed runner; rendering is owned by its clock."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

from .camera_calibration import build_inverse_map, remap_rgb_depth
from ..core.overlap_labels import rotation, OverlapRecorder


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class ThreeCameraWriter:
    def __init__(self, run_dir, definitions, scene):
        self.root = Path(run_dir)
        self.definitions = definitions
        self.sensors = {name: scene[d['scene_key']] for name, d in definitions.items()}
        # The outer capture clock owns the sampling cadence. SensorBase.update
        # force_recompute only refreshes *outdated* buffers, and its float32
        # period clock can suppress a requested sample (including a short final
        # interval). Always make a capture-time read eligible for refresh.
        self.original_periods = {}
        for name, sensor in self.sensors.items():
            self.original_periods[name] = float(sensor.cfg.update_period)
            sensor.cfg.update_period = 0.0
        self.maps = {}
        metadata = self.root / 'metadata'
        metadata.mkdir(parents=True, exist_ok=True)
        calibration = {}
        for name, d in definitions.items():
            cal = d['render_calibration']
            width, height = cal['image_size']
            *maps, diagnostics = build_inverse_map(cal, width, height)
            self.maps[name] = maps
            calibration[name] = dict(d, projection_mapping=diagnostics,
                original_sensor_update_period_s=self.original_periods[name],
                effective_sensor_update_period_s=0.0,
                sampling_clock='outer_capture_clock; fresh sensor read on every write')
        with (metadata / 'three_camera_calibration.json').open('x') as stream:
            json.dump(dict(cameras=calibration, depth_units='metres',
                depth_definition='distance_to_camera', invalid_depth='positive_infinity',
                base_quaternion_order='wxyz',
                pose_boundary='rigid mount model; native readback diagnostic, not pixel synchronization proof'), stream, indent=2)
        self.stream = (metadata / 'frame_index.jsonl').open('x')
        self.count = 0
        self.overlap_recorders = {}
        self.closed = False

    def write(self, timestamp, step_index, position, quaternion, overlap_labeler=None, overlap_input=None):
        row = dict(frame_index=self.count, timestamp_s=float(timestamp), step_index=int(step_index),
                   base_position_world=np.asarray(position).tolist(),
                   base_quaternion_wxyz_world=np.asarray(quaternion).tolist(),
                   modalities={}, camera_world_transforms={}, overlap_by_camera={})
        for name, sensor in self.sensors.items():
            sensor.update(0.0, force_recompute=True)
            output = sensor.data.output
            rgb = as_numpy(output['rgb'][0])[..., :3]
            depth = as_numpy(output['distance_to_camera'][0])
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            rgb, depth = remap_rgb_depth(rgb, depth, *self.maps[name])
            paths = {}
            for kind, suffix in [('rgb', 'png'), ('depth', 'npy')]:
                folder = self.root / 'captures' / kind / name
                folder.mkdir(parents=True, exist_ok=True)
                paths[kind] = folder / f'frame_{self.count:04d}.{suffix}'
                if paths[kind].exists():
                    raise FileExistsError(paths[kind])
            Image.fromarray(rgb, mode='RGB').save(paths['rgb'])
            np.save(paths['depth'], depth.astype(np.float32))
            row['modalities'][name] = dict(rgb=str(paths['rgb'].relative_to(self.root)),
                depth_float32=str(paths['depth'].relative_to(self.root)))
            definition = self.definitions[name]
            point = np.asarray(position) + rotation(quaternion) @ np.asarray(definition['mount_position_vehicle_xyz_m'])
            transform = np.eye(4)
            transform[:3, :3] = rotation(quaternion) @ rotation(definition['mount_quaternion_wxyz'])
            transform[:3, 3] = point
            row['camera_world_transforms'][name] = dict(matrix=transform.tolist(),
                camera_axes_convention=definition['mount_convention'],
                native_position_diagnostic=as_numpy(sensor.data.pos_w[0]).tolist())
            row['overlap_by_camera'][name] = (
                overlap_labeler.evaluate(overlap_input, point, 'rigid_mount_full_root_pose_at_capture')
                if overlap_labeler is not None and overlap_input is not None
                else dict(camera_evaluated=False))
            if row['overlap_by_camera'][name]['camera_evaluated']:
                if name not in self.overlap_recorders:
                    self.overlap_recorders[name] = OverlapRecorder(self.root/'metadata/overlap_by_camera'/name)
                recorder = self.overlap_recorders[name]
                label = row['overlap_by_camera'][name]
                recorder.observe(label)
                recorder.frame(label, self.count, timestamp_s=timestamp, camera_id=name)
        self.stream.write(json.dumps(row) + '\n')
        self.stream.flush()
        self.count += 1

    def close(self, completed=False):
        if self.closed:
            return
        for recorder in self.overlap_recorders.values():
            recorder.close(completed=completed)
        self.stream.close()
        self.closed = True
