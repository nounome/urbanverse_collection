"""Quantitative matching between automatic and reviewed obstacle OBBs."""

from __future__ import annotations

from typing import Any

import numpy as np


def _corners(row: dict[str, Any]) -> np.ndarray:
    value = row.get("corners_xy", row.get("world_obb_corners_xy"))
    points = np.asarray(value, dtype=np.float32)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError(f"obstacle is not a finite four-corner OBB: {row.get('path', row.get('id'))}")
    centre = points.mean(axis=0)
    hull = points[np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))]
    if _area(hull) <= 0.0:
        raise ValueError(f"obstacle OBB is degenerate: {row.get('path', row.get('id'))}")
    return hull


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1)))


def _area(points: np.ndarray) -> float:
    return abs(_signed_area(points))


def _convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping for the two convex four-corner OBBs."""
    clip = clip if _signed_area(clip) > 0.0 else clip[::-1]
    output = [np.asarray(point, dtype=np.float64) for point in subject]
    for left, right in zip(clip, np.roll(clip, -1, axis=0)):
        source = output
        output = []
        if not source:
            break

        def cross2(a, b):
            return float(a[0] * b[1] - a[1] * b[0])

        def inside(point):
            return cross2(right - left, point - left) >= -1.0e-9

        def intersection(a, b):
            segment = b - a
            edge = right - left
            denominator = cross2(segment, edge)
            if abs(denominator) <= 1.0e-12:
                return b
            t = cross2(left - a, edge) / denominator
            return a + t * segment

        previous = source[-1]
        for current in source:
            if inside(current):
                if not inside(previous):
                    output.append(intersection(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersection(previous, current))
            previous = current
    return np.asarray(output, dtype=np.float64)


def obb_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = _corners(left), _corners(right)
    area_a, area_b = _area(a), _area(b)
    clipped = _convex_intersection(a, b)
    intersection = _area(clipped) if len(clipped) >= 3 else 0.0
    union = area_a + area_b - intersection
    return min(1.0, max(0.0, intersection / union)) if union > 1.0e-12 else 0.0


def _identifier(row: dict[str, Any], index: int, prefix: str) -> str:
    return str(row.get("stable_id", row.get("id", row.get("path", f"{prefix}_{index:03d}"))))


def match_obstacle_inventories(
    candidates: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    *,
    minimum_iou: float = 0.20,
) -> dict[str, Any]:
    """Greedily select one-to-one maximum-IoU matches and expose every miss/FP."""

    pairs = []
    invalid_candidates = []
    invalid_reviewed = []
    valid_candidate_indices = []
    valid_reviewed_indices = []
    for index, row in enumerate(candidates):
        try:
            _corners(row)
            valid_candidate_indices.append(index)
        except ValueError as exc:
            invalid_candidates.append({"id": _identifier(row, index, "candidate"), "reason": str(exc)})
    for index, row in enumerate(reviewed):
        try:
            _corners(row)
            valid_reviewed_indices.append(index)
        except ValueError as exc:
            invalid_reviewed.append({"id": _identifier(row, index, "reviewed"), "reason": str(exc)})
    for ci in valid_candidate_indices:
        candidate = candidates[ci]
        for ri in valid_reviewed_indices:
            reference = reviewed[ri]
            score = obb_iou(candidate, reference)
            if score >= minimum_iou:
                pairs.append((score, ci, ri))
    pairs.sort(reverse=True)
    used_candidates: set[int] = set()
    used_reviewed: set[int] = set()
    matches = []
    for score, ci, ri in pairs:
        if ci in used_candidates or ri in used_reviewed:
            continue
        used_candidates.add(ci)
        used_reviewed.add(ri)
        matches.append(
            {
                "candidate_id": _identifier(candidates[ci], ci, "candidate"),
                "reviewed_id": _identifier(reviewed[ri], ri, "reviewed"),
                "iou": round(score, 6),
            }
        )
    unmatched_candidates = [
        _identifier(row, index, "candidate")
        for index, row in enumerate(candidates)
        if index not in used_candidates
    ]
    unmatched_reviewed = [
        _identifier(row, index, "reviewed")
        for index, row in enumerate(reviewed)
        if index not in used_reviewed
    ]
    tp = len(matches)
    fp = len(unmatched_candidates)
    fn = len(unmatched_reviewed)
    recall = tp / len(reviewed) if reviewed else (1.0 if not candidates else 0.0)
    precision = tp / len(candidates) if candidates else (1.0 if not reviewed else 0.0)
    return {
        "schema_version": 1,
        "method": "one-to-one greedy maximum convex OBB IoU",
        "minimum_iou": float(minimum_iou),
        "candidate_count": len(candidates),
        "reviewed_count": len(reviewed),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "matches": matches,
        "unmatched_candidate_ids": unmatched_candidates,
        "unmatched_reviewed_ids": unmatched_reviewed,
        "invalid_candidate_geometry": invalid_candidates,
        "invalid_reviewed_geometry": invalid_reviewed,
        "automatic_disposition": "needs_review" if fp or fn else "high_confidence_match",
        "candidate_deletion_performed": False,
    }
