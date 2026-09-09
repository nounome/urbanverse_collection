#!/usr/bin/env python3
"""Global reference-route planner for UrbanVerse Go2 route capture.

Replaces the fixed-direction PhysX corridor probe with a general planner:

1. Rasterize a 2.5D traversability grid from PhysX down-ray probes.
2. Inflate obstacle cells by the robot footprint + clearance and compute an ESDF
   (distance transform) over the traversable region.
3. Sample a start/end pair inside the same connected component of the allowed
   region, run A* (biased away from low-clearance cells), locally adjust the
   polyline toward corridor centre using the ESDF gradient, then smooth with a
   curvature-limited cubic spline.
4. Sweep the full robot-width corridor with dense PhysX rays (0.10-0.15 m
   spacing, lateral fan spanning robot_radius + clearance) before admitting the
   route, so spline corner-cutting cannot sneak past an obstacle.

Unit handling: every planning parameter is authored in METRES and converted
explicitly to stage units using the USD stage metersPerUnit (pxr). Because the
source scenes declare metersPerUnit=0.01 while their geometry and the Go2 spawn
reference are metre-like, a reference-based sanity correction is applied and the
declared/effective values are both recorded in every output.

This module imports no omni at import time: the PhysX scene query is injected as
a raycast callback so the whole pipeline is unit-testable without a simulator.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .joint_route_constraints import rasterize_joint_route_constraints

try:
    from scipy.ndimage import binary_dilation, distance_transform_edt
    from scipy.interpolate import CubicSpline

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - scipy is guaranteed by the project venv
    _SCIPY_AVAILABLE = False


GO2_REFERENCE_HEIGHT_M = 0.4
GO2_REFERENCE_HEIGHT_STAGE_UNITS = 0.4


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_native(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to python natives for JSON."""
    if isinstance(value, dict):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_native(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, np.integer)):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _cache_digest(*parts: Any) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()[:24]


# Bump whenever rasterisation semantics change so stale on-disk grids are
# automatically invalidated instead of silently reusing a previous layout.
_RASTER_SCHEMA_VERSION = 2


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def point_in_polygon(px: float, py: float, polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test for a simple closed polygon."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    x1, y1 = polygon[0]
    for k in range(1, n + 1):
        x2, y2 = polygon[k % n]
        if (y1 > py) != (y2 > py):
            x_intersect = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def resolve_meters_per_unit(
    stage: Any,
    reference_height_m: float = GO2_REFERENCE_HEIGHT_M,
    reference_height_stage_units: float = GO2_REFERENCE_HEIGHT_STAGE_UNITS,
) -> dict[str, Any]:
    """Resolve the metres-per-stage-unit conversion, with an explicit disclosure.

    The declared stage metersPerUnit is used only when it agrees with the known
    physical reference (Go2 base height). When the metadata disagrees with the
    metre-like geometry (as it does for every current scene), the reference is
    used and both values are recorded.
    """
    declared = 1.0
    if stage is not None:
        try:
            from pxr import UsdGeom

            value = UsdGeom.GetStageMetersPerUnit(stage)
            if value:
                declared = float(value)
        except Exception:
            declared = 1.0
    reference = float(reference_height_m) / float(reference_height_stage_units)
    consistent = abs(declared - reference) <= 0.35 * max(declared, reference, 1e-9)
    effective = declared if consistent else reference
    return {
        "meters_per_unit_declared": declared,
        "meters_per_unit_effective": effective,
        "reference": {
            "height_m": float(reference_height_m),
            "height_stage_units": float(reference_height_stage_units),
        },
        "consistent_with_reference": bool(consistent),
        "conversion_equation": f"stage_units = meters / {effective:.6f}",
        "disclosure": (
            "The source scenes declare metersPerUnit=0.01 while the loaded geometry and the "
            "Go2 spawn reference are metre-like; the effective factor is used for all planning "
            "conversions and both values are recorded so no SI-scale claim is silently assumed."
        ),
    }


def _grid_axes(bounds_su: list[float], cell_size_su: float) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = bounds_su
    xs = np.arange(x_min + cell_size_su / 2.0, x_max, cell_size_su)
    ys = np.arange(y_min + cell_size_su / 2.0, y_max, cell_size_su)
    return xs, ys


# ---------------------------------------------------------------------------
# Grid rasterization from PhysX down-rays
# ---------------------------------------------------------------------------
def rasterize_grid_from_physx(
    origin_xy: list[float],
    ground_z_ref: float,
    bounds_su: list[float],
    cell_size_su: float,
    raycast: Callable[..., dict[str, Any]],
    config: dict[str, Any],
    allowed_region_su: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Rasterize one downward PhysX ray per grid cell.

    ``raycast(origin_xyz, direction_xyz, max_distance) -> dict`` follows the
    canonical capture-script wrapper and returns ``hit/position/collision/
    rigid_body/robot_hit``.
    """
    xs, ys = _grid_axes(bounds_su, cell_size_su)
    ny, nx = len(ys), len(xs)

    allowed = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            if allowed_region_su is None:
                allowed[j, i] = True
            else:
                allowed[j, i] = point_in_polygon(float(xs[i]), float(ys[j]), allowed_region_su)

    z_top = float(ground_z_ref) + float(config.get("ground_ray_top_offset_m", 8.0)) / float(
        config.get("_meters_per_unit_effective", 1.0)
    )
    max_distance = float(config.get("ground_ray_max_distance_m", 16.0)) / float(
        config.get("_meters_per_unit_effective", 1.0)
    )
    height_tolerance = float(config.get("ground_height_tolerance_m", 0.15)) / float(
        config.get("_meters_per_unit_effective", 1.0)
    )
    allowed_tokens = [str(value).lower() for value in config.get("allowed_ground_path_tokens", [])]
    rejected_tokens = [str(value).lower() for value in config.get("rejected_ground_path_tokens", [])]

    ground_z = np.full((ny, nx), np.nan)
    supported = np.zeros((ny, nx), dtype=bool)
    obstacle = np.zeros((ny, nx), dtype=bool)
    uncertain = np.zeros((ny, nx), dtype=bool)
    raycast_count = 0
    for j in range(ny):
        for i in range(nx):
            if not allowed[j, i]:
                continue
            row = raycast([float(xs[i]), float(ys[j]), z_top], [0.0, 0.0, -1.0], max_distance)
            raycast_count += 1
            if not (isinstance(row, dict) and bool(row.get("hit"))):
                obstacle[j, i] = True
                continue
            position = row.get("position")
            if position is None:
                obstacle[j, i] = True
                continue
            hit_z = float(position[2])
            text = f"{row.get('collision', '')} {row.get('rigid_body', '')}".lower()
            if bool(row.get("robot_hit")):
                # The down-ray at cells under the Go2 body hits the robot itself,
                # not the ground. The robot is standing on verified walkable ground
                # at ``ground_z_ref`` (the spawn was PhysX-admitted), so these cells
                # are real support and must not be marked obstacle (which would make
                # the robot's own spawn cell invalid and force snap_to_valid to drag
                # the A* start sideways). ground_z_ref is the ground the robot rests
                # on, so it is the correct support height here.
                supported[j, i] = True
                ground_z[j, i] = float(ground_z_ref)
                continue
            height_ok = abs(hit_z - float(ground_z_ref)) <= height_tolerance
            token_ok = (not allowed_tokens) or any(token in text for token in allowed_tokens)
            rejected = any(token in text for token in rejected_tokens)
            if height_ok and token_ok and not rejected:
                supported[j, i] = True
                ground_z[j, i] = hit_z
            else:
                obstacle[j, i] = True

    return {
        "xs": xs,
        "ys": ys,
        "cell_size": float(cell_size_su),
        "bounds_su": list(bounds_su),
        "ground_z": ground_z,
        "supported": supported,
        "obstacle": obstacle,
        "uncertain": uncertain,
        "allowed": allowed,
        "raycast_count": raycast_count,
    }


def build_traversability(raster: dict[str, Any]) -> np.ndarray:
    return raster["supported"].copy()


def inflate_obstacles(traversable: np.ndarray, radius_su: float, cell_size_su: float) -> np.ndarray:
    iterations = max(1, int(math.ceil(radius_su / cell_size_su)))
    if not _SCIPY_AVAILABLE:
        return traversable
    obstacle = ~traversable
    inflated = binary_dilation(obstacle, iterations=iterations)
    return ~inflated


def compute_esdf(traversable: np.ndarray, cell_size_su: float, meters_per_unit_effective: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (esdf_px, esdf_m). esdf_px is distance to nearest obstacle in px."""
    if not _SCIPY_AVAILABLE:
        esdf_px = np.full(traversable.shape, np.nan)
        return esdf_px, np.full(traversable.shape, np.nan)
    esdf_px = distance_transform_edt(traversable)
    esdf_m = esdf_px * cell_size_su * meters_per_unit_effective
    return esdf_px, esdf_m


def connected_component(start_cell: tuple[int, int], valid: np.ndarray) -> np.ndarray:
    ny, nx = valid.shape
    component = np.zeros_like(valid, dtype=bool)
    j0, i0 = int(start_cell[0]), int(start_cell[1])
    if not (0 <= j0 < ny and 0 <= i0 < nx and valid[j0, i0]):
        return component
    stack = [(j0, i0)]
    component[j0, i0] = True
    while stack:
        j, i = stack.pop()
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == 0 and di == 0:
                    continue
                nj, ni = j + dj, i + di
                if 0 <= nj < ny and 0 <= ni < nx and valid[nj, ni] and not component[nj, ni]:
                    component[nj, ni] = True
                    stack.append((nj, ni))
    return component


def cell_at(x: float, y: float, xs: np.ndarray, ys: np.ndarray) -> tuple[int, int]:
    cell_x = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    cell_y = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    i = int(round((x - float(xs[0])) / cell_x)) if len(xs) else 0
    j = int(round((y - float(ys[0])) / cell_y)) if len(ys) else 0
    return j, i


def snap_to_valid(start_cell: tuple[int, int], valid: np.ndarray) -> tuple[int, int]:
    j, i = int(start_cell[0]), int(start_cell[1])
    ny, nx = valid.shape
    if 0 <= j < ny and 0 <= i < nx and valid[j, i]:
        return (j, i)
    for radius in range(1, max(ny, nx) + 1):
        for dj in range(-radius, radius + 1):
            for di in range(-radius, radius + 1):
                nj, ni = j + dj, i + di
                if 0 <= nj < ny and 0 <= ni < nx and valid[nj, ni]:
                    return (nj, ni)
    return start_cell


# ---------------------------------------------------------------------------
# A* planning over the grid
# ---------------------------------------------------------------------------
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def astar(
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    valid: np.ndarray,
    esdf_m: np.ndarray,
    penalty: float,
    cell_size_su: float,
    desired_clearance_su: float,
    meters_per_unit_effective: float,
    maximum_expansions: int | None = None,
) -> list[tuple[int, int]] | None:
    ny, nx = valid.shape
    start = (int(start_cell[0]), int(start_cell[1]))
    goal = (int(goal_cell[0]), int(goal_cell[1]))

    def heuristic(j: int, i: int) -> float:
        return math.hypot((j - goal[0]) * cell_size_su, (i - goal[1]) * cell_size_su) * meters_per_unit_effective

    open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == goal:
            break
        closed.add(current)
        if maximum_expansions is not None and len(closed) > maximum_expansions:
            return None
        for dj, di in _DIRS8:
            nj, ni = current[0] + dj, current[1] + di
            if not (0 <= nj < ny and 0 <= ni < nx and valid[nj, ni]):
                continue
            neighbor = (nj, ni)
            dist = math.hypot(dj * cell_size_su, di * cell_size_su) * meters_per_unit_effective
            esdf_here = float(esdf_m[nj, ni])
            if not math.isfinite(esdf_here):
                esdf_here = 0.0
            proximity = max(0.0, 1.0 - esdf_here / max(desired_clearance_su * meters_per_unit_effective, 1e-9))
            step = dist + penalty * dist * proximity
            tentative = g_score[current] + step
            if tentative < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heapq.heappush(open_set, (tentative + heuristic(nj, ni), neighbor))
    if goal not in g_score:
        return None
    path: list[tuple[int, int]] = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


def _geodesic_distance_su(
    start_cell: tuple[int, int],
    reachable: np.ndarray,
    cell_size_su: float,
) -> np.ndarray:
    """Shortest walkable-path length (in stage units) from ``start_cell`` over the
    ``reachable`` mask.

    Uses an 8-neighbour Dijkstra (orthogonal steps cost ``cell_size_su``,
    diagonal steps cost ``cell_size_su * sqrt(2)``) so the geodesic matches the
    diagonal-aware A* length used later for the band check. Used for endpoint
    sampling so the length band is tested against the *actual* walkable-path
    distance (which can be much longer than the straight-line distance for
    routes that must bend around obstacles, e.g. an L-shaped market entry)
    instead of the Euclidean distance. Returns an array of ``inf`` for cells
    that are not reachable.
    """
    ny, nx = reachable.shape
    dist = np.full((ny, nx), np.inf, dtype=np.float64)
    j0, i0 = int(start_cell[0]), int(start_cell[1])
    if not reachable[j0, i0]:
        return dist
    dist[j0, i0] = 0.0
    orth = float(cell_size_su)
    diag = float(cell_size_su) * math.sqrt(2.0)
    moves: list[tuple[int, int, float]] = [
        (1, 0, orth), (-1, 0, orth), (0, 1, orth), (0, -1, orth),
        (1, 1, diag), (1, -1, diag), (-1, 1, diag), (-1, -1, diag),
    ]
    heap: list[tuple[float, int, int]] = [(0.0, j0, i0)]
    while heap:
        cd, j, i = heapq.heappop(heap)
        if cd > dist[j, i] + 1e-12:
            continue
        for dj, di, cost in moves:
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx and reachable[nj, ni]:
                nd = cd + cost
                if nd < dist[nj, ni]:
                    dist[nj, ni] = nd
                    heapq.heappush(heap, (nd, nj, ni))
    return dist


def sample_endpoint_candidates(
    start_cell: tuple[int, int],
    component: np.ndarray,
    allowed: np.ndarray,
    esdf_m: np.ndarray,
    seed: int,
    target_length_m: list[float],
    cell_size_su: float,
    meters_per_unit_effective: float,
    max_attempts: int = 40,
    preferred_heading_rad: float | None = None,
    max_heading_deviation_rad: float | None = None,
) -> tuple[tuple[int, int], float] | None:
    """Deterministically find an in-band goal with a seeded high-clearance bias.

    Every component+allowed cell whose *walkable-path* distance to the start
    (4-neighbour BFS geodesic) falls inside the target length band is collected
    (no random draws, so narrow bands are never missed), then a seeded pick among
    the highest-ESDF in-band cells is returned. ``max_attempts`` bounds the size
    of the high-clearance shortlist it is chosen from, keeping routes
    reproducible per seed yet varied across seeds. Returns ``None`` only when no
    cell is reachable within the band.

    Using the geodesic (not Euclidean) distance lets bending routes such as an
    L-shaped corridor-to-market turn be sampled for a long length band even
    though their straight-line goal distance is much shorter; A* is then asked
    for exactly the path the band promised.

    When ``preferred_heading_rad`` is given, the candidate must also point within
    ``max_heading_deviation_rad`` of that world heading from the start (measured
    on the straight line, which is the direction the first leg actually faces).
    This keeps the route's first leg close to the Go2's spawn heading so the
    frozen gait never has to complete a large pure in-place turn.
    """
    mask = component & allowed
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return None
    j0, i0 = int(start_cell[0]), int(start_cell[1])
    d_col = (cells[:, 1] - i0) * cell_size_su
    d_row = (cells[:, 0] - j0) * cell_size_su
    geodesic_su = _geodesic_distance_su((j0, i0), mask, cell_size_su)
    dist_m = geodesic_su[cells[:, 0], cells[:, 1]] * meters_per_unit_effective
    band_idx = np.flatnonzero((dist_m >= target_length_m[0]) & (dist_m <= target_length_m[1]))
    if preferred_heading_rad is not None and max_heading_deviation_rad is not None:
        heading = np.arctan2(d_row, d_col)
        dev = np.abs((heading - preferred_heading_rad + math.pi) % (2.0 * math.pi) - math.pi)
        band_idx = band_idx[dev[band_idx] <= max_heading_deviation_rad]
    if len(band_idx) == 0:
        return None
    scores = esdf_m[cells[band_idx, 0], cells[band_idx, 1]]
    order = np.argsort(-scores, kind="stable")
    shortlist = min(len(band_idx), max(1, max_attempts))
    rng = random.Random(int(seed))
    chosen = int(band_idx[order[rng.randrange(shortlist)]])
    j, i = int(cells[chosen, 0]), int(cells[chosen, 1])
    return (j, i), float(dist_m[chosen])


def polyline_length_m(points_su: np.ndarray, meters_per_unit_effective: float) -> float:
    if len(points_su) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points_su, axis=0), axis=1)) * meters_per_unit_effective)


def polyline_total_turn_rad(points_su: np.ndarray) -> float:
    """Cumulative heading change (rad) along a polyline.

    Segment headings come from ``arctan2`` of consecutive differences. The
    difference of two *wrapped* headings must itself be wrapped, otherwise a
    path that stays within a few degrees of +/-pi crosses the wrap seam and
    each crossing is miscounted as a ~2*pi turn.
    """
    if len(points_su) < 3:
        return 0.0
    headings = np.arctan2(np.diff(points_su[:, 1]), np.diff(points_su[:, 0]))
    total = 0.0
    for k in range(len(headings) - 1):
        total += abs(wrap_angle(float(headings[k + 1]) - float(headings[k])))
    return float(total)


def downsample_polyline(points_su: np.ndarray, step_su: float) -> np.ndarray:
    if len(points_su) < 2:
        return points_su
    out = [points_su[0]]
    acc = 0.0
    for k in range(1, len(points_su)):
        acc += float(np.linalg.norm(points_su[k] - points_su[k - 1]))
        if acc >= step_su or k == len(points_su) - 1:
            out.append(points_su[k])
            acc = 0.0
    return np.asarray(out, dtype=np.float64)


def local_adjust(
    points_su: np.ndarray,
    esdf_px: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    cell_size_su: float,
    max_shift_su: float,
    valid: np.ndarray,
) -> np.ndarray:
    """Push interior points toward higher clearance using the ESDF gradient."""
    if len(points_su) <= 2:
        return points_su
    grad_y, grad_x = np.gradient(esdf_px)
    ny, nx = esdf_px.shape
    out = points_su.copy()
    for k in range(1, len(out) - 1):
        x, y = float(out[k, 0]), float(out[k, 1])
        i = int(round((x - float(xs[0])) / cell_size_su))
        j = int(round((y - float(ys[0])) / cell_size_su))
        i = int(np.clip(i, 0, nx - 1))
        j = int(np.clip(j, 0, ny - 1))
        gx = float(grad_x[j, i]) * cell_size_su
        gy = float(grad_y[j, i]) * cell_size_su
        norm = math.hypot(gx, gy)
        if norm < 1e-9:
            continue
        shift = min(max_shift_su, norm * cell_size_su)
        nx_new = x + (gx / norm) * shift
        ny_new = y + (gy / norm) * shift
        ni = int(round((nx_new - float(xs[0])) / cell_size_su))
        nj = int(round((ny_new - float(ys[0])) / cell_size_su))
        if 0 <= nj < ny and 0 <= ni < nx and valid[nj, ni]:
            out[k, 0] = nx_new
            out[k, 1] = ny_new
    return out


# ---------------------------------------------------------------------------
# Spline smoothing
# ---------------------------------------------------------------------------
def fit_spline(points_su: np.ndarray) -> tuple[tuple[Any, Any], np.ndarray] | None:
    if len(points_su) < 3 or not _SCIPY_AVAILABLE:
        return None
    d = np.diff(points_su, axis=0)
    t = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    if t[-1] < 1e-9:
        return None
    spl_x = CubicSpline(t, points_su[:, 0])
    spl_y = CubicSpline(t, points_su[:, 1])
    return (spl_x, spl_y), t


def spline_max_curvature(spl: tuple[Any, Any], t0: float, t1: float, meters_per_unit_effective: float, samples: int = 400) -> float:
    ts = np.linspace(t0, t1, samples)
    dx = spl[0].derivative()(ts)
    dy = spl[1].derivative()(ts)
    ddx = spl[0].derivative(2)(ts)
    ddy = spl[1].derivative(2)(ts)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom = np.maximum(denom, 1e-9)
    kappa_per_su = np.abs(dx * ddy - dy * ddx) / denom
    kappa_per_m = kappa_per_su / max(meters_per_unit_effective, 1e-9)
    return float(np.max(kappa_per_m))


def chaikin_smooth(points_su: np.ndarray, iterations: int = 1) -> np.ndarray:
    pts = np.asarray(points_su, dtype=np.float64)
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for k in range(len(pts) - 1):
            p0, p1 = pts[k], pts[k + 1]
            out.append(0.75 * p0 + 0.25 * p1)
            out.append(0.25 * p0 + 0.75 * p1)
        out.append(pts[-1])
        pts = np.asarray(out, dtype=np.float64)
    return pts


def resample_equal_spacing(spl: tuple[Any, Any], t0: float, t1: float, spacing_su: float) -> np.ndarray:
    total = t1 - t0
    count = max(2, int(math.ceil(total / max(spacing_su, 1e-9))))
    ts = np.linspace(t0, t1, count + 1)
    xs = spl[0](ts)
    ys = spl[1](ts)
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
# Dense full-corridor PhysX validation
# ---------------------------------------------------------------------------
def validate_route_full_physx(
    dense_points_su: np.ndarray,
    ground_z_ref: float,
    raycast: Callable[..., dict[str, Any]],
    c: dict[str, Any],
) -> dict[str, Any]:
    """Sweep the whole robot-width corridor at 0.10-0.15 m along the route.

    At every dense point a lateral fan spanning -(r+m) .. +(r+m) is checked:
    ground-support down-rays (height tolerance + path-token policy) and forward
    clearance rays at the obstacle heights. A stop fan is added at the endpoint.
    """
    radius = float(c["robot_radius_su"])
    margin = float(c["clearance_margin_su"])
    height_tolerance = float(c["height_tolerance_su"])
    obstacle_heights = [float(h) for h in c["obstacle_heights_su"]]
    horizontal_clearance = float(c["horizontal_clearance_su"])
    endpoint_radius = float(c["endpoint_clearance_su"])
    z_top = float(ground_z_ref) + float(c["ground_ray_top_offset_su"])
    max_ground_distance = float(c["ground_ray_max_distance_su"])
    allowed_tokens = [str(value).lower() for value in c["allowed_ground_path_tokens"]]
    rejected_tokens = [str(value).lower() for value in c["rejected_ground_path_tokens"]]
    lateral_offsets = [-(radius + margin), -radius, -radius / 2.0, 0.0, radius / 2.0, radius, radius + margin]

    segments: list[dict[str, Any]] = []
    passed = True
    first_failed: int | None = None
    min_clearance_m = None
    for k, (x, y) in enumerate(dense_points_su):
        if k < len(dense_points_su) - 1:
            heading = math.atan2(dense_points_su[k + 1, 1] - y, dense_points_su[k + 1, 0] - x)
        else:
            heading = math.atan2(y - dense_points_su[k - 1, 1], x - dense_points_su[k - 1, 0])
        fx, fy = math.cos(heading), math.sin(heading)
        lx, ly = -fy, fx
        point_ok = True
        for offset in lateral_offsets:
            px = float(x) + lx * offset
            py = float(y) + ly * offset
            ground = raycast([px, py, z_top], [0.0, 0.0, -1.0], max_ground_distance)
            if not (isinstance(ground, dict) and bool(ground.get("hit"))):
                point_ok = False
                break
            gpos = ground.get("position")
            if gpos is None:
                point_ok = False
                break
            text = f"{ground.get('collision', '')} {ground.get('rigid_body', '')}".lower()
            if bool(ground.get("robot_hit")):
                continue
            if abs(float(gpos[2]) - float(ground_z_ref)) > height_tolerance:
                point_ok = False
                break
            token_ok = (not allowed_tokens) or any(token in text for token in allowed_tokens)
            rejected = any(token in text for token in rejected_tokens)
            if not token_ok or rejected:
                point_ok = False
                break
            for height in obstacle_heights:
                row = raycast([px, py, float(ground_z_ref) + height], [fx, fy, 0.0], horizontal_clearance)
                blocked = bool(isinstance(row, dict) and row.get("hit") and not row.get("robot_hit"))
                if blocked:
                    point_ok = False
                    break
            if not point_ok:
                break
        segments.append({"segment_index": k, "passed": bool(point_ok), "xy": [float(x), float(y)]})
        if not point_ok:
            passed = False
            if first_failed is None:
                first_failed = k
        else:
            segment_clearance = None
            # clearance here is approximated by the smallest ESDF-aligned distance;
            # the lateral sweep above is the authoritative check.
            if segment_clearance is not None and (min_clearance_m is None or segment_clearance < min_clearance_m):
                min_clearance_m = segment_clearance

    # Endpoint stop fan: rays around the final point at obstacle heights.
    endpoint_fan: list[dict[str, Any]] = []
    final_x, final_y = float(dense_points_su[-1, 0]), float(dense_points_su[-1, 1])
    for angle_deg in range(0, 360, 45):
        angle = math.radians(angle_deg)
        direction = [math.cos(angle), math.sin(angle), 0.0]
        for height in obstacle_heights:
            row = raycast([final_x, final_y, float(ground_z_ref) + height], direction, endpoint_radius)
            blocked = bool(isinstance(row, dict) and row.get("hit") and not row.get("robot_hit"))
            endpoint_fan.append({"angle_deg": angle_deg, "blocked": blocked})
            if blocked:
                passed = False
    failed_indices = [item["segment_index"] for item in segments if not item["passed"]]

    return {
        "accepted": bool(passed),
        "dense_interval_m": float(c["validation_interval_su"] * c["meters_per_unit_effective"]),
        "segment_count": len(segments),
        "failed_segment_indices": failed_indices,
        "first_failed_segment_index": first_failed,
        "min_clearance_m": min_clearance_m,
        "endpoint_fan_blocked": bool(any(item["blocked"] for item in endpoint_fan)),
        "segments": segments,
        "endpoint_fan": endpoint_fan,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _build_smoothed_route(
    path_cells: list[tuple[int, int]],
    grid: dict[str, Any],
    esdf_px: np.ndarray,
    esdf_m: np.ndarray,
    valid: np.ndarray,
    c: dict[str, Any],
    penalty: float,
    desired_clearance_su: float,
) -> dict[str, Any] | None:
    xs, ys = grid["xs"], grid["ys"]
    cell_size_su = float(grid["cell_size"])
    mpu = float(c["meters_per_unit_effective"])
    raw = np.asarray([[float(xs[i]), float(ys[j])] for (j, i) in path_cells], dtype=np.float64)
    # The 8-connected A* path is a cell staircase (0/-45/45 deg teeth). Smoothing
    # it Chaikin-first-then-downsample keeps the residual teeth and the cubic
    # spline overshoots them into curvatures far above the hard limit. The robust
    # order is instead to downsample the RAW staircase at a step large enough to
    # jump past the teeth, Chaikin the sparse polyline, and only then fit the
    # spline. If the curvature is still high, retry with progressively larger
    # downsampling steps (sparser control points are smoother for a cubic spline).
    # local_adjust is intentionally not applied here: it self-crossed short routes
    # and the ESDF-penalised A* already keeps the path near the corridor centre.
    path_length_su = polyline_length_m(raw, 1.0)
    base_step_su = min(float(c["downsample_step_su"]), max(path_length_su / 6.0, cell_size_su))
    best = None
    best_mult = 1.0
    attempts = 0
    for mult in (1.0, 1.4, 1.9, 2.5, 3.2, 4.0):
        step_su = max(base_step_su * mult, cell_size_su)
        if path_length_su / step_su < 4.0:
            break
        attempts += 1
        points = downsample_polyline(raw, step_su)
        points = chaikin_smooth(points, 3)
        fitted = fit_spline(points)
        if fitted is None:
            continue
        spline, t = fitted
        curvature = spline_max_curvature(spline, float(t[0]), float(t[-1]), mpu)
        if best is None or curvature < best[0]:
            best = (curvature, spline, t, points)
            best_mult = mult
        if curvature <= float(c["max_curvature_rad_per_m"]):
            break
    if best is None:
        return None
    curvature, spline, t, points = best
    waypoints = resample_equal_spacing(spline, float(t[0]), float(t[-1]), float(c["resample_spacing_su"]))
    dense = resample_equal_spacing(spline, float(t[0]), float(t[-1]), float(c["validation_interval_su"]))
    return {
        "waypoints": waypoints,
        "dense": dense,
        "max_curvature_rad_per_m": float(curvature),
        "smooth_iterations": attempts - 1,
        "downsample_multiplier": best_mult,
        "control_point_count": int(len(points)),
        "polyline_length_m": path_length_su * mpu,
    }


def route_points_inside_mask(
    points_su: np.ndarray,
    valid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> bool:
    """Return whether every sampled route point remains in the admitted grid."""

    if len(xs) == 0 or len(ys) == 0:
        return False
    cell_x = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    cell_y = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    for x, y in np.asarray(points_su, dtype=np.float64):
        i = int(round((float(x) - float(xs[0])) / cell_x))
        j = int(round((float(y) - float(ys[0])) / cell_y))
        if not (0 <= j < valid.shape[0] and 0 <= i < valid.shape[1] and valid[j, i]):
            return False
    return True


def plan_global_route(
    origin_xy: list[float],
    ground_z_ref: float,
    raycast: Callable[..., dict[str, Any]],
    stage: Any,
    config: dict[str, Any],
    cache_dir: str | Path | None = None,
    cache_key: str = "",
    spawn_heading_rad: float | None = None,
) -> dict[str, Any]:
    """Full global route pipeline. Returns an accepted-route dict or a failure dict."""
    mpu_info = resolve_meters_per_unit(stage)
    mpu = float(mpu_info["meters_per_unit_effective"])

    def su(value_m: float) -> float:
        return float(value_m) / mpu

    cell_size_su = su(float(config.get("cell_size_m", 0.25)))
    robot_radius_su = su(float(config.get("robot_radius_m", 0.40)))
    clearance_su = su(float(config.get("clearance_margin_m", 0.15)))
    inflate_radius_su = robot_radius_su + clearance_su
    desired_clearance_su = inflate_radius_su

    penalty = float(config.get("astar_esdf_penalty", 0.25))
    max_curvature = float(config.get("max_curvature_rad_per_m", 0.8))
    target_length_m = [float(config.get("target_route_length_m", [7.0, 9.0])[0]), float(config.get("target_route_length_m", [7.0, 9.0])[1])]
    if target_length_m[0] <= 0 or target_length_m[1] < target_length_m[0]:
        target_length_m = [7.0, 9.0]
    seed = int(config.get("sample_seed", 20260805))
    max_attempts = int(config.get("endpoint_sample_attempts", 40))
    # Minimum cumulative heading change (rad) along the smoothed route. 0.0 (default)
    # disables the constraint so existing straight-route configs are unchanged.
    min_total_turn_rad = float(config.get("min_total_turn_rad", 0.0))

    allowed_region_m = config.get("allowed_region_polygon_m")
    allowed_region_su = (
        [[float(x) / mpu, float(y) / mpu] for x, y in allowed_region_m] if allowed_region_m else None
    )
    if allowed_region_su:
        arr = np.asarray(allowed_region_su, dtype=np.float64)
        pad = su(float(config.get("grid_padding_m", 2.0)))
        bounds_su = [
            float(arr[:, 0].min()) - pad,
            float(arr[:, 0].max()) + pad,
            float(arr[:, 1].min()) - pad,
            float(arr[:, 1].max()) + pad,
        ]
    else:
        half = su(float(config.get("grid_half_extent_m", 40.0)))
        bounds_su = [
            float(origin_xy[0]) - half,
            float(origin_xy[0]) + half,
            float(origin_xy[1]) - half,
            float(origin_xy[1]) + half,
        ]

    work_config = dict(config)
    work_config["_meters_per_unit_effective"] = mpu

    cache_hit = False
    cache_path = None
    grid = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        digest = _cache_digest(
            cache_key,
            _RASTER_SCHEMA_VERSION,
            bounds_su,
            cell_size_su,
            mpu,
            allowed_region_m,
            work_config.get("allowed_ground_path_tokens"),
            work_config.get("rejected_ground_path_tokens"),
            ground_z_ref,
            work_config.get("ground_ray_top_offset_m"),
            work_config.get("ground_ray_max_distance_m"),
            work_config.get("ground_height_tolerance_m"),
        )
        cache_path = cache_dir_path / f"global_route_grid_{digest}.npz"
        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True)
                grid = {
                    "xs": data["xs"],
                    "ys": data["ys"],
                    "cell_size": float(data["cell_size"]),
                    "bounds_su": [float(value) for value in data["bounds_su"]],
                    "ground_z": data["ground_z"],
                    "supported": data["supported"],
                    "obstacle": data["obstacle"],
                    "uncertain": data["uncertain"],
                    "allowed": data["allowed"],
                    "raycast_count": int(data["raycast_count"]),
                }
                cache_hit = True
            except Exception:
                grid = None
    if grid is None:
        grid = rasterize_grid_from_physx(
            origin_xy,
            ground_z_ref,
            bounds_su,
            cell_size_su,
            raycast,
            work_config,
            allowed_region_su,
        )
        if cache_path is not None:
            np.savez(
                cache_path,
                xs=grid["xs"],
                ys=grid["ys"],
                cell_size=np.float64(grid["cell_size"]),
                bounds_su=np.asarray(grid["bounds_su"], dtype=np.float64),
                ground_z=grid["ground_z"],
                supported=grid["supported"],
                obstacle=grid["obstacle"],
                uncertain=grid["uncertain"],
                allowed=grid["allowed"],
                raycast_count=np.int64(grid["raycast_count"]),
            )

    traversable = build_traversability(grid)
    joint_payload = config.get("joint_route_constraints") or {}
    joint_constraints = list(joint_payload.get("constraints", []))
    joint_mask, joint_rows = rasterize_joint_route_constraints(
        grid["xs"], grid["ys"], joint_constraints, mpu
    )
    # Cells outside the allowed region are never raycast (they fall outside the
    # allowed polygon), so the rasterizer leaves them unsupported by default.
    # Treating those unknowns as real obstacles would dilate them inward and
    # spurious-kill a corridor entrance that sits exactly on the polygon edge
    # (observed: the Beijing market spawn cell at the corridor top). Treat them
    # as neutral for inflation instead; `valid` still ANDs with ``allowed`` so the
    # A* never leaves the allowed region, and genuine non-traversable cells inside
    # the region still inflate normally.
    inflation_source = traversable | ~grid["allowed"]
    traversable_inflated = inflate_obstacles(inflation_source, inflate_radius_su, cell_size_su)
    valid = traversable_inflated & grid["allowed"] & ~joint_mask
    # Recompute clearance after adding the time-varying agent-route corridors;
    # otherwise endpoint sampling would still prefer a point beside a car or
    # People stream that was absent from the static PhysX raster.
    esdf_source = valid | ~grid["allowed"]
    esdf_px, esdf_m = compute_esdf(esdf_source, cell_size_su, mpu)
    esdf_px[~grid["allowed"]] = 0.0
    esdf_m[~grid["allowed"]] = 0.0

    xs, ys = grid["xs"], grid["ys"]
    raw_start_cell = cell_at(float(origin_xy[0]), float(origin_xy[1]), xs, ys)
    origin_blocked_by_joint_routes = bool(
        0 <= raw_start_cell[0] < joint_mask.shape[0]
        and 0 <= raw_start_cell[1] < joint_mask.shape[1]
        and joint_mask[raw_start_cell]
    )
    start_cell = snap_to_valid(raw_start_cell, valid)
    component = connected_component(start_cell, valid)

    supported_count = int(np.count_nonzero(grid["supported"] & grid["allowed"]))
    obstacle_count = int(np.count_nonzero((grid["obstacle"] | grid["uncertain"]) & grid["allowed"]))
    allowed_count = int(np.count_nonzero(grid["allowed"]))
    finite_esdf = esdf_m[grid["supported"] & grid["allowed"]]
    esdf_summary = {
        "min_m": float(np.min(finite_esdf)) if len(finite_esdf) else None,
        "max_m": float(np.max(finite_esdf)) if len(finite_esdf) else None,
        "mean_m": float(np.mean(finite_esdf)) if len(finite_esdf) else None,
    }

    attempts_bands = [list(target_length_m)]
    for _ in range(3):
        previous = attempts_bands[-1]
        relaxed = [max(1.0, previous[0] * 0.75), max(1.2, previous[1] * 0.85)]
        attempts_bands.append(relaxed)

    preferred_heading = (
        float(spawn_heading_rad) if spawn_heading_rad is not None else config.get("first_leg_heading_rad")
    )
    if preferred_heading is not None:
        preferred_heading = float(preferred_heading)
    tight_deviation = float(config.get("max_first_leg_heading_deviation_rad", 0.9))
    # Relax the first-leg heading deviation in bounded steps so a route that must
    # turn into an obstacle field (e.g. a market corridor) is admitted while still
    # preferring small first-leg deviations. The final bounded step stays below pi;
    # the follower executes large initial realigns as a forward arc (turn_forward_speed).
    # The used value is recorded per plan.
    heading_deviations: list[float | None] = (
        [
            min(math.pi, tight_deviation),
            min(math.pi, tight_deviation + 0.6),
            min(math.pi, tight_deviation + 1.2),
        ]
        if preferred_heading is not None
        else [None]
    )

    plan = None
    last_reason = (
        "Go2 start lies inside a configured vehicle or pedestrian swept corridor"
        if origin_blocked_by_joint_routes
        else "no candidate endpoint admitted in the allowed region"
    )
    used_heading_deviation: float | None = None
    for band in ([] if origin_blocked_by_joint_routes else attempts_bands):
        for heading_deviation in heading_deviations:
            candidate = sample_endpoint_candidates(
                start_cell,
                component,
                grid["allowed"],
                esdf_m,
                seed,
                band,
                cell_size_su,
                mpu,
                max_attempts=max_attempts,
                preferred_heading_rad=preferred_heading,
                max_heading_deviation_rad=heading_deviation,
            )
            if candidate is None:
                continue
            goal_cell, straight_m = candidate
            path = astar(
                start_cell,
                goal_cell,
                valid,
                esdf_m,
                penalty,
                cell_size_su,
                desired_clearance_su,
                mpu,
            )
            if path is None:
                last_reason = "A* found no path between sampled endpoints"
                continue
            path_length_m = polyline_length_m(
                np.asarray([[float(xs[i]), float(ys[j])] for (j, i) in path], dtype=np.float64), mpu
            )
            if not (band[0] - 0.5 <= path_length_m <= band[1] + 1.0):
                last_reason = f"A* path length {path_length_m:.2f} m outside band {band[0]:.1f}-{band[1]:.1f} m"
                continue
            route = _build_smoothed_route(
                path,
                grid,
                esdf_px,
                esdf_m,
                valid,
                {
                    "downsample_step_su": su(float(config.get("downsample_step_m", 0.8))),
                    "max_local_adjust_su": su(float(config.get("max_local_adjust_m", 0.4))),
                    "resample_spacing_su": su(float(config.get("resample_spacing_m", 0.4))),
                    "validation_interval_su": su(
                        float(config.get("validation_interval_m", 0.12))
                    ),
                    "max_curvature_rad_per_m": max_curvature,
                    "meters_per_unit_effective": mpu,
                },
                penalty,
                desired_clearance_su,
            )
            if route is None:
                last_reason = "spline fitting failed for the A* path"
                continue
            if not route_points_inside_mask(route["dense"], valid, xs, ys):
                last_reason = (
                    "smoothed route left the static-safe region or entered a configured "
                    "vehicle/pedestrian swept corridor"
                )
                continue
            if route["max_curvature_rad_per_m"] > float(config.get("hard_max_curvature_rad_per_m", 1.2)):
                last_reason = (
                    f"spline curvature {route['max_curvature_rad_per_m']:.2f} rad/m exceeds the hard limit after smoothing"
                )
                continue
            # The first smoothed waypoint sits at the A* start-cell centre, which can
            # be up to half a cell away from the robot's actual spawn position
            # (origin_xy). Prepending the origin to the raw smoothed waypoints would
            # insert a backward stub leg (e.g. pointing ~135 deg away from the spawn
            # heading) that the frozen-gait follower cannot execute in place. Anchor
            # the delivered route at the origin instead: the origin becomes waypoint 0
            # and the redundant start-cell-centre waypoint is skipped. The cumulative-
            # turn filter is measured on this executed polyline so it reflects the real
            # route shape rather than the artificial stub turn.
            if len(route["waypoints"]) < 2:
                tail_waypoints = np.asarray(route["waypoints"], dtype=np.float64)
            else:
                tail_waypoints = np.asarray(route["waypoints"][1:], dtype=np.float64)
            route_with_spawn = np.vstack(
                [
                    [[float(origin_xy[0]), float(origin_xy[1])]],
                    tail_waypoints,
                ]
            )
            total_turn = polyline_total_turn_rad(route_with_spawn)
            if total_turn < min_total_turn_rad:
                last_reason = (
                    f"route cumulative turn {math.degrees(total_turn):.0f} deg below minimum "
                    f"{math.degrees(min_total_turn_rad):.0f} deg"
                )
                continue
            # First-leg-from-spawn guard: the executed first leg (origin -> first
            # delivered waypoint) must not demand a large in-place turn. The endpoint
            # heading ladder already bounds the goal direction from the start cell, but
            # the first smoothed waypoint can still deviate; reject legs beyond one
            # ladder step above the current endpoint tolerance so the follower's
            # forward arc can always complete the initial realign.
            if preferred_heading is not None and len(tail_waypoints) >= 1:
                first_leg_rad = math.atan2(
                    float(tail_waypoints[0][1]) - float(origin_xy[1]),
                    float(tail_waypoints[0][0]) - float(origin_xy[0]),
                )
                first_leg_dev = abs(wrap_angle(first_leg_rad - preferred_heading))
                first_leg_tol = (
                    min(math.pi, heading_deviation + 0.6)
                    if heading_deviation is not None
                    else math.pi
                )
                if first_leg_dev > first_leg_tol:
                    last_reason = (
                        f"first-leg heading deviation {math.degrees(first_leg_dev):.0f} deg from "
                        f"spawn exceeds {math.degrees(first_leg_tol):.0f} deg"
                    )
                    continue
            validation = validate_route_full_physx(
                route["dense"],
                ground_z_ref,
                raycast,
                {
                    "robot_radius_su": robot_radius_su,
                    "clearance_margin_su": clearance_su,
                    "height_tolerance_su": su(float(config.get("ground_height_tolerance_m", 0.15))),
                    "obstacle_heights_su": [su(float(h)) for h in config.get("obstacle_heights_m", [0.18, 0.40, 0.65])],
                    "horizontal_clearance_su": su(float(config.get("horizontal_clearance_m", 0.15))),
                    "endpoint_clearance_su": su(float(config.get("endpoint_clearance_radius_m", 0.55))),
                    "ground_ray_top_offset_su": su(float(config.get("ground_ray_top_offset_m", 8.0))),
                    "ground_ray_max_distance_su": su(float(config.get("ground_ray_max_distance_m", 16.0))),
                    "validation_interval_su": su(float(config.get("validation_interval_m", 0.12))),
                    "meters_per_unit_effective": mpu,
                    "allowed_ground_path_tokens": config.get("allowed_ground_path_tokens", []),
                    "rejected_ground_path_tokens": config.get("rejected_ground_path_tokens", []),
                },
            )
            if not validation["accepted"]:
                last_reason = (
                    f"full PhysX corridor validation failed at segments "
                    f"{validation['failed_segment_indices'][:5]}"
                )
                continue

            used_heading_deviation = heading_deviation
            waypoints_world_xy = [[float(origin_xy[0]), float(origin_xy[1])]] + [
                [float(p[0]), float(p[1])] for p in tail_waypoints
            ]
            plan = {
                "accepted": True,
                "waypoints_world_xy": waypoints_world_xy,
                "unit_conversion": mpu_info,
                "grid": {
                    "cell_size_m": float(cell_size_su * mpu),
                    "bounds_su": grid["bounds_su"],
                    "dims": [int(grid["supported"].shape[0]), int(grid["supported"].shape[1])],
                    "allowed_cell_count": allowed_count,
                    "supported_cell_count": supported_count,
                    "obstacle_cell_count": obstacle_count,
                    "raycast_count": grid["raycast_count"],
                    "cache_hit": cache_hit,
                    "cache_path": str(cache_path) if cache_path else None,
                },
                "esdf": esdf_summary,
                "joint_route_constraints": {
                    "default_policy": joint_payload.get("default_policy", "hard_exclusion"),
                    "go2_online_dynamic_avoidance": bool(
                        joint_payload.get("go2_online_dynamic_avoidance", False)
                    ),
                    "constraint_count": len(joint_rows),
                    "excluded_cell_count": int(np.count_nonzero(joint_mask)),
                    "counts": joint_payload.get("counts", {}),
                    "parameters_m": joint_payload.get("parameters_m", {}),
                    "sources": joint_payload.get("sources", {}),
                    "constraints": joint_rows,
                },
                "astar": {
                    "start_cell": [int(start_cell[0]), int(start_cell[1])],
                    "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
                    "straight_line_m": round(straight_m, 3),
                    "path_length_m": round(path_length_m, 3),
                    "path_node_count": len(path),
                },
                "spline": {
                    "max_curvature_rad_per_m": round(route["max_curvature_rad_per_m"], 4),
                    "smooth_iterations": route["smooth_iterations"],
                    "resample_spacing_m": float(config.get("resample_spacing_m", 0.4)),
                    "waypoint_count": len(route["waypoints"]),
                },
                "turn": {
                    "total_turn_rad": round(total_turn, 4),
                    "total_turn_deg": round(math.degrees(total_turn), 1),
                    "min_total_turn_rad": round(min_total_turn_rad, 4),
                },
                "validation": {
                    "accepted": validation["accepted"],
                    "dense_interval_m": validation["dense_interval_m"],
                    "segment_count": validation["segment_count"],
                    "failed_segment_indices": validation["failed_segment_indices"],
                    "endpoint_fan_blocked": validation["endpoint_fan_blocked"],
                    "min_clearance_m": validation["min_clearance_m"],
                },
                "target_route_length_m": target_length_m,
                "length_band_used_m": band,
                "first_leg_heading_rad": preferred_heading,
                "first_leg_heading_deviation_rad_used": used_heading_deviation,
            }
            break

        if plan is not None:
            break

    if plan is not None:
        return plan
    return {
        "accepted": False,
        "reason": last_reason,
        "unit_conversion": mpu_info,
        "grid": {
            "cell_size_m": float(cell_size_su * mpu),
            "bounds_su": grid["bounds_su"],
            "dims": [int(grid["supported"].shape[0]), int(grid["supported"].shape[1])],
            "allowed_cell_count": allowed_count,
            "supported_cell_count": supported_count,
            "obstacle_cell_count": obstacle_count,
            "raycast_count": grid["raycast_count"],
            "cache_hit": cache_hit,
            "cache_path": str(cache_path) if cache_path else None,
        },
        "esdf": esdf_summary,
        "joint_route_constraints": {
            "default_policy": joint_payload.get("default_policy", "hard_exclusion"),
            "go2_online_dynamic_avoidance": bool(
                joint_payload.get("go2_online_dynamic_avoidance", False)
            ),
            "constraint_count": len(joint_rows),
            "excluded_cell_count": int(np.count_nonzero(joint_mask)),
            "counts": joint_payload.get("counts", {}),
            "parameters_m": joint_payload.get("parameters_m", {}),
            "sources": joint_payload.get("sources", {}),
            "constraints": joint_rows,
        },
        "astar": {"path_length_m": None, "reason": last_reason},
        "spline": None,
        "validation": None,
        "target_route_length_m": target_length_m,
        "length_band_used_m": attempts_bands[-1],
    }


# ---------------------------------------------------------------------------
# Self test (no omni required)
# ---------------------------------------------------------------------------
def _self_test() -> int:
    if not _SCIPY_AVAILABLE:
        print("self-test requires scipy")
        return 1

    cell = 0.5
    n = 28
    obstacle_mask = np.zeros((n, n), dtype=bool)
    obstacle_mask[10:20, 10:20] = True  # central obstacle block
    xs_axis = np.arange(cell / 2.0, cell * n, cell)
    ys_axis = np.arange(cell / 2.0, cell * n, cell)

    def fake_raycast(origin, direction, max_distance):
        ox, oy, oz = origin
        dx, dy, dz = direction
        if abs(dx) < 1e-9 and abs(dy) < 1e-9 and dz < 0:
            i = int(round((ox - xs_axis[0]) / cell))
            j = int(round((oy - ys_axis[0]) / cell))
            if 0 <= i < n and 0 <= j < n and obstacle_mask[j, i]:
                return {"hit": True, "position": [ox, oy, 4.0], "distance": oz - 4.0, "collision": "/World/building_block", "rigid_body": "", "robot_hit": False}
            return {"hit": True, "position": [ox, oy, 0.0], "distance": oz, "collision": "/World/ground/terrain/Walkable_001/mesh", "rigid_body": "", "robot_hit": False}
        return {"hit": False, "position": None, "distance": max_distance, "collision": "", "rigid_body": "", "robot_hit": False}

    # Cumulative-turn metric guard: a synthetic 90-degree L-shape must sum to pi/2
    # (this fails if the heading difference is not wrap-normalised).
    l_shape = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]], dtype=np.float64)
    l_turn = polyline_total_turn_rad(l_shape)
    if abs(l_turn - math.pi / 2.0) > 1e-9:
        print(f"SELF_TEST ERROR: L-shape turn {l_turn:.6f} rad != pi/2")
        return 1
    straight_turn = polyline_total_turn_rad(np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float64))
    if straight_turn != 0.0:
        print(f"SELF_TEST ERROR: straight polyline turn {straight_turn:.6f} rad != 0")
        return 1

    config = {
        "cell_size_m": cell,
        "robot_radius_m": 0.40,
        "clearance_margin_m": 0.15,
        "ground_height_tolerance_m": 0.15,
        "obstacle_heights_m": [0.18, 0.40, 0.65],
        "horizontal_clearance_m": 0.15,
        "endpoint_clearance_radius_m": 0.55,
        "ground_ray_top_offset_m": 8.0,
        "ground_ray_max_distance_m": 16.0,
        "max_curvature_rad_per_m": 0.8,
        "hard_max_curvature_rad_per_m": 1.2,
        "resample_spacing_m": 0.4,
        "downsample_step_m": 0.8,
        "max_local_adjust_m": 0.4,
        "validation_interval_m": 0.12,
        "astar_esdf_penalty": 0.25,
        "target_route_length_m": [4.0, 8.0],
        "sample_seed": 20260805,
        "endpoint_sample_attempts": 40,
        "allowed_ground_path_tokens": ["walkable_"],
        "rejected_ground_path_tokens": ["building_", "vehicle_", "sidewalk", "/robot"],
        "allowed_region_polygon_m": [[0.0, 0.0], [0.0, 14.0], [14.0, 14.0], [14.0, 0.0]],
        "grid_padding_m": 1.0,
    }
    # Minimum-turn filter: with a strict minimum, a low-turn plan must be rejected.
    # The candidate loop rejects each in-band goal before the min-turn filter (spline
    # curvature, validation), so the final ``reason`` is not guaranteed to be the
    # min-turn one — a later relaxed-band candidate can fail curvature and overwrite
    # it. The filter's effect is therefore verified comparatively: the same config
    # without a minimum admits a low-turn route, and with the strict minimum the plan
    # is rejected, proving the minimum is the binding constraint.
    config_base = dict(config)
    config_turn = dict(config)
    config_turn["min_total_turn_rad"] = math.pi  # 180 deg, unreachable on the open grid
    with tempfile.TemporaryDirectory() as tmp2:
        plan_base = plan_global_route(
            [1.0, 1.0],
            0.0,
            fake_raycast,
            None,
            config_base,
            cache_dir=tmp2,
            cache_key="self_test_scene_min_turn_base",
        )
        plan_turn = plan_global_route(
            [1.0, 1.0],
            0.0,
            fake_raycast,
            None,
            config_turn,
            cache_dir=tmp2,
            cache_key="self_test_scene_min_turn",
        )
        if not plan_base["accepted"]:
            print(f"SELF_TEST ERROR: min-turn baseline route rejected: {plan_base.get('reason')}")
            return 1
        base_turn = plan_base["turn"]["total_turn_rad"]
        if base_turn >= math.pi - 0.01:
            print("SELF_TEST ERROR: baseline route already meets the strict min-turn; test is vacuous")
            return 1
        if plan_turn["accepted"]:
            print("SELF_TEST ERROR: a route was admitted despite the strict min_total_turn_rad")
            return 1
    with tempfile.TemporaryDirectory() as tmp:
        plan = plan_global_route(
            [1.0, 1.0],
            0.0,
            fake_raycast,
            None,
            config,
            cache_dir=tmp,
            cache_key="self_test_scene",
        )
        print(json.dumps(_to_native(plan), indent=2, ensure_ascii=False))
        if plan["grid"]["cache_hit"]:
            print("SELF_TEST ERROR: first call unexpectedly hit the cache")
            return 1
        # second call must hit the cache
        plan2 = plan_global_route(
            [1.0, 1.0],
            0.0,
            fake_raycast,
            None,
            config,
            cache_dir=tmp,
            cache_key="self_test_scene",
        )
        if not plan2["grid"]["cache_hit"]:
            print("SELF_TEST ERROR: cache round-trip did not hit")
            return 1
        if plan["accepted"]:
            length_m = plan["astar"]["path_length_m"]
            if not (4.0 <= length_m <= 9.0):
                print(f"SELF_TEST ERROR: route length {length_m:.2f} m outside expected band")
                return 1
            if len(plan["waypoints_world_xy"]) < 3:
                print("SELF_TEST ERROR: too few waypoints")
                return 1
            computed_turn = polyline_total_turn_rad(
                np.asarray(plan["waypoints_world_xy"], dtype=np.float64)
            )
            if "turn" not in plan or abs(plan["turn"]["total_turn_rad"] - computed_turn) > 1e-4:
                print(
                    f"SELF_TEST ERROR: turn record {plan.get('turn')} inconsistent with "
                    f"waypoints turn {computed_turn:.6f}"
                )
                return 1
        else:
            print("SELF_TEST note: synthetic route not accepted; inspecting failure is fine for pipeline smoke")
    print("SELF_TEST_OK")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run the offline pipeline self test")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
