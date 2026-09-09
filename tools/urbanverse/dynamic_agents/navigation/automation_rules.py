"""Scene-independent planning rules; proposals are not runtime qualification."""
from __future__ import annotations

import math
import numpy as np


def coverage_metrics(points, cell_m=1.0):
    """Distinct visited grid cells, not enclosed area or repeated lap length.

    Resample every segment so a sparse long path cannot lose to a dense small
    loop. Grid origin is the world origin, independent of scene bounds.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("route requires finite Nx2 points")
    if not math.isfinite(cell_m) or cell_m <= 0:
        raise ValueError("coverage cell size must be positive")
    cells = set()
    length = 0.0
    for a, b in zip(p[:-1], p[1:]):
        segment = float(np.linalg.norm(b - a))
        length += segment
        if segment / cell_m > 100000:
            raise ValueError("segment exceeds coverage sampling budget")
        samples = np.linspace(a, b, max(2, math.ceil(segment / (cell_m / 4)) + 1))
        cells.update(map(tuple, np.floor(samples / cell_m).astype(np.int64)))
    return dict(unique_coverage_cells=len(cells), coverage_cell_m=float(cell_m),
                sampled_coverage_area_m2=len(cells) * cell_m**2, length_m=length)


def loop_rank(row, cell_m=1.0):
    metrics = coverage_metrics(row['xy'], cell_m)
    row['coverage_metrics'] = metrics
    # Coverage first; fewer repetitions win at identical coverage. Length is
    # positive only after equal repetition efficiency. No global optimum claim.
    efficiency = metrics['unique_coverage_cells'] * cell_m / max(metrics['length_m'], 1e-9)
    return metrics['unique_coverage_cells'], efficiency, metrics['length_m']


def allocate_capacity(area_m2, route_lengths_m, *, area_per_agent_m2,
                      spacing_m, maximum_count):
    """Area AND route-length cap. Spawn OBB checks remain mandatory downstream."""
    values = [area_m2, area_per_agent_m2, spacing_m, *route_lengths_m]
    if not all(math.isfinite(v) for v in values) or area_m2 < 0:
        raise ValueError('nonfinite/negative capacity input')
    if area_per_agent_m2 <= 0 or spacing_m <= 0 or any(v < 0 for v in route_lengths_m):
        raise ValueError('invalid density/spacing/length')
    if isinstance(maximum_count, bool) or int(maximum_count) != maximum_count or maximum_count < 0:
        raise ValueError('maximum_count must be a nonnegative integer')
    per_route = [int(v // spacing_m) for v in route_lengths_m]
    count = min(int(area_m2 // area_per_agent_m2), sum(per_route), int(maximum_count))
    assigned = [0] * len(per_route)
    for _ in range(count):
        options = [i for i, cap in enumerate(per_route) if assigned[i] < cap]
        i = max(options, key=lambda j: route_lengths_m[j] / (assigned[j] + 1))
        assigned[i] += 1
    return dict(count=count, per_route=assigned, area_m2=float(area_m2),
                area_per_agent_m2=float(area_per_agent_m2), spacing_m=float(spacing_m),
                status='proposed_requires_spawn_and_motion_checks' if count else 'no_capacity')


def shared_capacity(area_m2, lengths, settings):
    """People and bikes spend ONE area budget; shares are explicit parameters."""
    names = ('people', 'micromobility')
    shares = [settings[n]['area_share'] for n in names]
    if not all(math.isfinite(s) and s >= 0 for s in shares) or sum(shares) > 1 + 1e-9:
        raise ValueError('shared area shares must be nonnegative and sum to <= 1')
    return {name: allocate_capacity(area_m2 * settings[name]['area_share'], lengths[name],
            **{k: settings[name][k] for k in ('area_per_agent_m2', 'spacing_m', 'maximum_count')})
            for name in names}


def decide_issue(code, *, core_affected, attempts_used, retry_limit):
    """Finite retry policy over explicit measured findings, not an image oracle."""
    if attempts_used < 0 or retry_limit < 0:
        raise ValueError('negative retry budget')
    if code == 'gpu_health_inconsistent':
        return dict(action='defer_launch', severity='environment', scene_failed=False)
    cosmetic = {'local_shadow_flicker', 'local_material_defect'}
    if (code in cosmetic and not core_affected) or code == 'go2_dynamic_overlap':
        return dict(action='continue_with_label', severity='warning', scene_failed=False)
    recoverable = {'exposure', 'camera_obstruction', 'route_candidate_failed',
                   'congestion', 'load_failed', 'support_failed', 'go2_unstable',
                   'capture_contract_failed', 'render_motion_unsynchronized'}
    if code not in recoverable and code not in cosmetic:
        return dict(action='needs_diagnosis', severity='unknown', scene_failed=False)
    retry = attempts_used < retry_limit
    return dict(action='retry_adjusted' if retry else 'stop_scene',
                severity='core' if core_affected else 'adjustable', scene_failed=not retry)


def select_distance_route(candidates, targets=(100., 50., 25.), tolerance_fraction=.1,
                          maximum_repeat_ratio=2.5):
    """Choose among already geometry-checked candidates; never extend by looping.

    Shorter tiers indicate bounded candidate-search failure, not proof that the
    scene physically cannot fit a larger route. Revalidate the returned prefix.
    """
    if not targets or not 0 <= tolerance_fraction < 1:
        raise ValueError('invalid route distance tiers/tolerance')
    if not math.isfinite(maximum_repeat_ratio) or maximum_repeat_ratio <= 0:
        raise ValueError('invalid repetition budget')
    if any(not math.isfinite(t) or t <= 0 for t in targets) or list(targets) != sorted(set(targets), reverse=True):
        raise ValueError('distance tiers must be positive, finite, unique and descending')
    for target in targets:
        choices = []
        for index, points in enumerate(candidates):
            p = np.asarray(points, dtype=float)
            coverage_metrics(p)  # finite/shape validation
            keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-8]
            p = p[keep]
            if len(p) < 3:
                continue
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            if arc[-1] < target * (1-tolerance_fraction):
                continue
            length = min(float(target), float(arc[-1]))
            prefix = p[arc < length]
            end = [np.interp(length, arc, p[:, axis]) for axis in (0, 1)]
            prefix = np.vstack([prefix, end])
            delta = np.diff(prefix, axis=0)
            headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
            turn = float(np.abs(np.diff(headings)).sum())
            coverage = coverage_metrics(prefix)
            if length > maximum_repeat_ratio * coverage['unique_coverage_cells'] * coverage['coverage_cell_m']:
                continue
            choices.append(((turn, coverage['unique_coverage_cells']), index, prefix))
        if choices:
            _, index, path = max(choices, key=lambda item: item[0])
            return path, dict(target_distance_m=float(target), candidate_index=index,
                actual_distance_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                larger_tiers_attempted=[float(t) for t in targets if t > target],
                selection_scope='best qualified candidate within search budget, not scene capacity proof')
    raise ValueError('No candidate meets any distance tier; do not fabricate repeated laps')
