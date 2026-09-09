"""Durable raster walking regions derived from road semantics and height obstacles.

The raster is an admission boundary, not a replacement for Recast NavMesh.  It
keeps random goals inside the approved NearRoad/NearBuffer surface while the
official navigation runtime performs the final path query and static-obstacle
avoidance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


SEMANTIC_COLOURS = {
    "nearroad": (66, 129, 201),
    "nearbuffer": (230, 184, 63),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Eight-connected labels without a scipy/opencv runtime dependency."""

    try:
        import cv2
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        return labels,[int(stats[i,cv2.CC_STAT_AREA]) for i in range(1,count)]
    except ImportError:
        pass
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    sizes: list[int] = []
    label = 0
    neighbours = tuple(
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dx != 0 or dy != 0
    )
    for row, column in zip(*np.nonzero(mask)):
        if labels[row, column] != 0:
            continue
        label += 1
        count = 0
        queue = deque(((int(row), int(column)),))
        labels[row, column] = label
        while queue:
            current_row, current_column = queue.popleft()
            count += 1
            for dy, dx in neighbours:
                next_row = current_row + dy
                next_column = current_column + dx
                if not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                if not mask[next_row, next_column] or labels[next_row, next_column] != 0:
                    continue
                labels[next_row, next_column] = label
                queue.append((next_row, next_column))
        sizes.append(count)
    return labels, sizes


@dataclass(frozen=True)
class WalkableRegionConfig:
    semantic_image: Path
    obstacle_mask: Path
    world_bounds_xyxy: tuple[float, float, float, float]
    allowed_semantics: tuple[str, ...]
    clearance_m: float
    minimum_component_area_m2: float
    ground_z_m: float
    fixed_seed: int
    erosion_metric: str = 'square'

    @classmethod
    def load(cls, path: Path) -> "WalkableRegionConfig":
        path = Path(path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        inputs = payload["inputs"]
        semantic = (path.parent / inputs["semantic_image"]).resolve()
        obstacle = (path.parent / inputs["obstacle_mask"]).resolve()
        for input_path in (semantic, obstacle):
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
        return cls(
            semantic_image=semantic,
            obstacle_mask=obstacle,
            world_bounds_xyxy=tuple(map(float, payload["world_bounds_xyxy"])),
            allowed_semantics=tuple(map(str.lower, payload["allowed_semantics"])),
            clearance_m=float(payload["clearance_m"]),
            minimum_component_area_m2=float(payload["minimum_component_area_m2"]),
            ground_z_m=float(payload["ground_z_m"]),
            fixed_seed=int(payload["fixed_seed"]),
            erosion_metric=str(payload.get('erosion_metric','square')),
        )


class WalkableRegions:
    """Connected, obstacle-eroded walking blocks with deterministic sampling."""

    def __init__(self, config: WalkableRegionConfig) -> None:
        semantic = np.asarray(Image.open(config.semantic_image).convert("RGB"), dtype=np.uint8)
        obstacles = np.asarray(Image.open(config.obstacle_mask).convert("L"), dtype=np.uint8) > 0
        if semantic.shape[:2] != obstacles.shape:
            raise ValueError(f"semantic/obstacle shape mismatch: {semantic.shape} vs {obstacles.shape}")
        allowed = np.zeros(obstacles.shape, dtype=bool)
        for name in config.allowed_semantics:
            if name not in SEMANTIC_COLOURS:
                raise ValueError(f"unsupported walking semantic: {name}")
            allowed |= np.all(semantic == np.asarray(SEMANTIC_COLOURS[name]), axis=2)
        admitted = allowed & ~obstacles
        x0, y0, x1, y1 = config.world_bounds_xyxy
        self.pixel_size_xy_m = ((x1 - x0) / admitted.shape[1], (y1 - y0) / admitted.shape[0])
        erosion_radius_px = int(np.ceil(config.clearance_m / min(self.pixel_size_xy_m)))
        if config.erosion_metric=='euclidean':
            from scipy.ndimage import distance_transform_edt
            admitted=admitted&(distance_transform_edt(admitted,sampling=self.pixel_size_xy_m[::-1])>=config.clearance_m)
        elif config.erosion_metric!='square':
            raise ValueError('Unknown erosion metric '+config.erosion_metric)
        elif erosion_radius_px > 0:
            size = 2 * erosion_radius_px + 1
            admitted = np.asarray(
                Image.fromarray(admitted.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(size))
            ) > 0
        labels, sizes = _label_components(admitted)
        pixel_area = self.pixel_size_xy_m[0] * self.pixel_size_xy_m[1]
        retained_old = [
            index + 1
            for index, size in enumerate(sizes)
            if size * pixel_area >= config.minimum_component_area_m2
        ]
        self.labels = np.zeros_like(labels)
        self.component_pixels: dict[int, np.ndarray] = {}
        for new_label, old_label in enumerate(retained_old, start=1):
            pixels = np.argwhere(labels == old_label)
            self.labels[labels == old_label] = new_label
            self.component_pixels[new_label] = pixels
        self.config = config
        self.allowed_mask = allowed
        self.obstacle_mask = obstacles
        self.admitted_mask = self.labels > 0

    @classmethod
    def load(cls, path: Path) -> "WalkableRegions":
        return cls(WalkableRegionConfig.load(path))

    @property
    def component_ids(self) -> tuple[int, ...]:
        return tuple(self.component_pixels)

    def semantic_subset(self, semantics: Iterable[str]) -> "WalkableRegions":
        """Rebuild the approved raster for one or more configured sub-bands.

        The subset reuses the same immutable semantic and obstacle evidence,
        metric clearance, world transform and component-area gate.  It is a
        portable admission operation rather than a Scene09 coordinate rule.
        """

        names = tuple(str(value).lower() for value in semantics)
        if not names:
            raise ValueError("semantic subset must be nonempty")
        unavailable = sorted(set(names) - set(self.config.allowed_semantics))
        if unavailable:
            raise ValueError(
                "semantic subset is outside the approved union: "
                + ", ".join(unavailable)
            )
        return WalkableRegions(replace(self.config, allowed_semantics=names))

    def component_maximum_width_m(self, component_id: int) -> float:
        """Estimate the widest traversable cross-section of one component.

        Repeated 3x3 erosion gives the Chebyshev in-radius without adding a
        scipy/OpenCV dependency.  It is deliberately conservative and is used
        only for population admission: a component narrower than the configured
        multi-person threshold may host one resident, but never a crowd.
        """

        component_id = int(component_id)
        if component_id not in self.component_pixels:
            raise KeyError(component_id)
        current = self.labels == component_id
        erosion_count = 0
        while np.any(current):
            erosion_count += 1
            current = np.asarray(
                Image.fromarray(current.astype(np.uint8) * 255).filter(
                    ImageFilter.MinFilter(3)
                )
            ) > 0
        return float(
            (2 * erosion_count - 1) * min(self.pixel_size_xy_m)
        )

    def pixel_to_world(self, row: int, column: int) -> np.ndarray:
        x0, _y0, _x1, y1 = self.config.world_bounds_xyxy
        pixel_x, pixel_y = self.pixel_size_xy_m
        return np.asarray(
            [x0 + (column + 0.5) * pixel_x, y1 - (row + 0.5) * pixel_y],
            dtype=np.float64,
        )

    def world_to_pixel(self, xy: Iterable[float]) -> tuple[int, int] | None:
        x, y = map(float, xy)
        x0, y0, x1, y1 = self.config.world_bounds_xyxy
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        column = min(self.labels.shape[1] - 1, int((x - x0) / (x1 - x0) * self.labels.shape[1]))
        row = min(self.labels.shape[0] - 1, int((y1 - y) / (y1 - y0) * self.labels.shape[0]))
        return row, column

    def component_at(self, xy: Iterable[float]) -> int:
        pixel = self.world_to_pixel(xy)
        return 0 if pixel is None else int(self.labels[pixel])

    def in_component_with_raster_tolerance(
        self,
        xy: Iterable[float],
        component_id: int,
        tolerance_m: float,
    ) -> bool:
        """Membership with an explicit sub-cell/raster boundary tolerance.

        Recast operates on continuous polygons while the approval artifact is
        a 30.6 cm raster in Scene09.  A character centre can therefore be on
        the approved polygon but quantize to the immediately adjacent pixel.
        This helper only accepts the same component within the declared metric
        tolerance; it does not bridge obstacles or another connected block.
        """
        pixel = self.world_to_pixel(xy)
        if pixel is None:
            return False
        if int(self.labels[pixel]) == int(component_id):
            return True
        radius_rows = int(np.ceil(tolerance_m / self.pixel_size_xy_m[1]))
        radius_columns = int(np.ceil(tolerance_m / self.pixel_size_xy_m[0]))
        row, column = pixel
        r0, r1 = max(0, row - radius_rows), min(self.labels.shape[0], row + radius_rows + 1)
        c0, c1 = max(0, column - radius_columns), min(self.labels.shape[1], column + radius_columns + 1)
        candidates = np.argwhere(self.labels[r0:r1, c0:c1] == int(component_id))
        for local_row, local_column in candidates:
            world = self.pixel_to_world(r0 + int(local_row), c0 + int(local_column))
            if float(np.linalg.norm(world - np.asarray(tuple(xy), dtype=np.float64))) <= tolerance_m:
                return True
        return False

    def is_admitted_with_raster_tolerance(
        self, xy: Iterable[float], tolerance_m: float
    ) -> bool:
        pixel = self.world_to_pixel(xy)
        if pixel is None:
            return False
        if int(self.labels[pixel]) != 0:
            return True
        radius_rows = int(np.ceil(tolerance_m / self.pixel_size_xy_m[1]))
        radius_columns = int(np.ceil(tolerance_m / self.pixel_size_xy_m[0]))
        row, column = pixel
        r0, r1 = max(0, row - radius_rows), min(
            self.labels.shape[0], row + radius_rows + 1
        )
        c0, c1 = max(0, column - radius_columns), min(
            self.labels.shape[1], column + radius_columns + 1
        )
        for local_row, local_column in np.argwhere(
            self.labels[r0:r1, c0:c1] != 0
        ):
            world = self.pixel_to_world(
                r0 + int(local_row), c0 + int(local_column)
            )
            if float(
                np.linalg.norm(world - np.asarray(tuple(xy), dtype=np.float64))
            ) <= tolerance_m:
                return True
        return False

    def contains_points(self, points_xy: np.ndarray, component_id: int | None = None) -> bool:
        points=np.asarray(points_xy,dtype=np.float64)
        if not len(points):return True
        x0,_,_,y1=self.config.world_bounds_xyxy
        columns=np.floor((points[:,0]-x0)/self.pixel_size_xy_m[0]).astype(int)
        rows=np.floor((y1-points[:,1])/self.pixel_size_xy_m[1]).astype(int)
        if not ((columns>=0)&(columns<self.labels.shape[1])&(rows>=0)&(rows<self.labels.shape[0])).all():return False
        labels=self.labels[rows,columns]
        return bool((labels>0).all() if component_id is None else (labels==component_id).all())

    def sample(self, component_id: int, rng: np.random.Generator) -> np.ndarray:
        pixels = self.component_pixels.get(int(component_id))
        if pixels is None or len(pixels) == 0:
            raise KeyError(f"unknown/empty walkable component: {component_id}")
        row, column = pixels[int(rng.integers(0, len(pixels)))]
        return self.pixel_to_world(int(row), int(column))

    def summary(self) -> dict[str, object]:
        pixel_area = self.pixel_size_xy_m[0] * self.pixel_size_xy_m[1]
        return {
            "world_bounds_xyxy": list(self.config.world_bounds_xyxy),
            "resolution_wh": [int(self.labels.shape[1]), int(self.labels.shape[0])],
            "pixel_size_xy_m": list(self.pixel_size_xy_m),
            "clearance_m": self.config.clearance_m,
            "allowed_semantics": list(self.config.allowed_semantics),
            "component_count": len(self.component_pixels),
            "components": [
                {
                    "component_id": component_id,
                    "pixel_count": int(len(pixels)),
                    "area_m2": float(len(pixels) * pixel_area),
                }
                for component_id, pixels in self.component_pixels.items()
            ],
        }
