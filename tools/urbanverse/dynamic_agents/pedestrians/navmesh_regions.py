"""Author a Recast-only mesh from approved raster walking blocks."""

from __future__ import annotations

from typing import Any, Sequence

from .walkable_regions import WalkableRegions


def _horizontal_runs(mask: Any):
    for row in range(mask.shape[0]):
        column = 0
        while column < mask.shape[1]:
            if not mask[row, column]:
                column += 1
                continue
            start = column
            while column < mask.shape[1] and mask[row, column]:
                column += 1
            yield row, start, column


def hide_scene_for_bake(
    stage: Any,
    *,
    scene_root_paths: Sequence[str] = ("/World",),
    keep_paths: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Temporarily hide original scene branches from the Recast bake.

    Recast in Isaac Sim 4.5 reads the composed stage each bake.  Excluding a
    top-level Xform with ``NavMeshExcludeAPI`` is not inherited into referenced
    payloads, so the only cheap whole-scene switch is render visibility.  The
    official smoke demonstrates that an *invisible* collider is not rasterized;
    hiding every pre-existing branch therefore leaves only the approved helper
    (and any explicitly kept prim) visible while baking.  The recorded
    visibility opinions are session-layer only; source USD files stay untouched.
    """

    from pxr import UsdGeom

    keep = set(keep_paths)
    records: list[dict[str, str]] = []
    for root_path in scene_root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(
                f"scene root is missing while hiding it for the bake: {root_path}"
            )
        # /World also owns runtime helpers, so hide its original children.  A
        # separately composed source-scene root can be hidden atomically.
        candidates = root.GetChildren() if root_path == "/World" else (root,)
        for candidate in candidates:
            candidate_path = str(candidate.GetPath())
            if candidate_path in keep or not candidate.IsA(UsdGeom.Imageable):
                continue
            imageable = UsdGeom.Imageable(candidate)
            original = imageable.GetVisibilityAttr().Get() or UsdGeom.Tokens.inherited
            if original == UsdGeom.Tokens.invisible:
                continue
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            records.append(
                {"path": candidate_path, "original_visibility": str(original)}
            )
    return records


def restore_scene_visibility(stage: Any, records: Sequence[dict[str, str]]) -> int:
    """Restore branch visibility that :func:`hide_scene_for_bake` changed."""

    from pxr import UsdGeom

    restored = 0
    for record in records:
        prim = stage.GetPrimAtPath(record["path"])
        if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            continue
        attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
        if record["original_visibility"] == str(UsdGeom.Tokens.inherited):
            attr.Clear()
        else:
            attr.Set(record["original_visibility"])
        restored += 1
    return restored


def author_approved_navmesh_surface(
    stage: Any,
    regions: WalkableRegions,
    *,
    mesh_path: str = "/World/ApprovedWalkingSurface",
    volume_path: str = "/World/ApprovedWalkingNavMeshVolumes",
    source_scene_path: str = "/World/ground/terrain",
    exclusion_strategy: str = "temporary_visibility",
) -> dict[str, object]:
    """Make only the approved blocks visible to Recast; hide the helper in RTX.

    Isaac Sim 4.5 has an exclusion API but no ``NavMeshIncludeAPI``, so Recast
    includes collision geometry by default.  In a composed UrbanVerse stage a
    nominally small bake traverses the entire city unless every other branch is
    removed from the Recast input.

    Two strategies are supported:

    - ``temporary_visibility`` (default): author the helper and tiled volumes,
      then set ``visibility = invisible`` on every pre-existing ``/World``
      branch.  No exclusion API is written to thousands of Gprims and no USD
      recomposition is triggered.  The caller must restore the recorded branch
      visibility through :func:`restore_scene_visibility` (or
      ``finalize_approved_navmesh_surface(..., scene_visibility_records=...)``)
      immediately after ``OFFICIAL_NAVMESH_READY``.
    - ``per_gprim_exclude``: the historical fallback that authors
      ``NavMeshExcludeAPI`` on every top-level Xform and every concrete Gprim.
      It is retained for A/B comparison only; the six prior runs proved it
      triggers expensive USD recomposition and still never reached a queryable
      NavMesh in bounded time.
    """

    import NavSchema
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    source = stage.GetPrimAtPath(source_scene_path)
    if not source.IsValid():
        raise RuntimeError(f"source scene prim is missing: {source_scene_path}")
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        raise RuntimeError("/World is missing while authoring the approved NavMesh")
    excluded_top_level_paths = []
    excluded_geometry_paths = []
    hidden_scene_branches: list[dict[str, str]] = []
    if exclusion_strategy == "per_gprim_exclude":
        for child in world.GetChildren():
            child_path = str(child.GetPath())
            if child_path in {mesh_path, volume_path}:
                continue
            if child.IsA(UsdGeom.Xformable):
                NavSchema.NavMeshExcludeAPI.Apply(child)
                excluded_top_level_paths.append(child_path)

        # NavMeshExcludeAPI is not inherited through arbitrary referenced scene
        # hierarchies in Isaac Sim 4.5.  Applying it only to /World/ground (or to
        # another ancestor Xform) therefore still lets Recast inspect thousands of
        # descendant meshes.  Author the API on every pre-existing concrete Gprim
        # as well.  Skip instance proxies because USD cannot author opinions on
        # them; their owning instance root is covered by the top-level exclusion.
        # This creates lightweight session-layer opinions and never edits the
        # source UrbanVerse USD.
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim_path == mesh_path or prim_path.startswith(f"{volume_path}/"):
                continue
            if prim.IsInstanceProxy():
                continue
            if prim.IsA(UsdGeom.Gprim):
                NavSchema.NavMeshExcludeAPI.Apply(prim)
                excluded_geometry_paths.append(prim_path)
    elif exclusion_strategy == "temporary_visibility":
        hidden_scene_branches = hide_scene_for_bake(
            stage,
            scene_root_paths=(
                ("/World",)
                if source_scene_path == "/World"
                else ("/World", source_scene_path)
            ),
            keep_paths=(mesh_path, volume_path),
        )
    else:
        raise ValueError(f"unsupported exclusion strategy: {exclusion_strategy}")

    x0, _y0, _x1, y1 = regions.config.world_bounds_xyxy
    pixel_x, pixel_y = regions.pixel_size_xy_m
    z = float(regions.config.ground_z_m) + 0.015
    points = []
    counts = []
    indices = []
    for row, start, end in _horizontal_runs(regions.admitted_mask):
        left = x0 + start * pixel_x
        right = x0 + end * pixel_x
        top = y1 - row * pixel_y
        bottom = y1 - (row + 1) * pixel_y
        offset = len(points)
        points.extend(
            (
                Gf.Vec3f(left, bottom, z),
                Gf.Vec3f(right, bottom, z),
                Gf.Vec3f(right, top, z),
                Gf.Vec3f(left, top, z),
            )
        )
        counts.append(4)
        indices.extend((offset, offset + 1, offset + 2, offset + 3))
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    # Recast 106.4 follows the same contract as NVIDIA's official smoke:
    # default-purpose, visible collision geometry while baking.  Making this
    # helper guide/invisible before baking causes get_navmesh() to remain None.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    # Walkable is the default area.  Applying the explicit area API makes the
    # intent inspectable without relying on a nonexistent 4.5 include API.
    NavSchema.NavMeshAreaAPI.Apply(mesh.GetPrim())

    # A single volume over the complete Scene09 bounds would voxelize roughly
    # 588 x 331 metres even though the admitted surface is only a set of thin
    # roadside strips.  Tile each connected component and overlap adjacent
    # volumes slightly; this preserves connectivity while bounding Recast work
    # by the admitted footprint rather than the scene AABB.
    stage.DefinePrim(volume_path, "Xform")
    volume_records = []
    tile_pixels = 64
    margin_m = 1.0
    for component_id, pixels in regions.component_pixels.items():
        tiles: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row_value, column_value in pixels:
            key = (int(row_value) // tile_pixels, int(column_value) // tile_pixels)
            tiles.setdefault(key, []).append((int(row_value), int(column_value)))
        for tile_index, tile_pixels_rc in enumerate(tiles.values()):
            values = list(zip(*tile_pixels_rc))
            row_min, row_max = min(values[0]), max(values[0])
            column_min, column_max = min(values[1]), max(values[1])
            left = max(x0, x0 + column_min * pixel_x - margin_m)
            right = min(_x1, x0 + (column_max + 1) * pixel_x + margin_m)
            top = min(y1, y1 - row_min * pixel_y + margin_m)
            bottom = max(_y0, y1 - (row_max + 1) * pixel_y - margin_m)
            child_path = f"{volume_path}/Component_{component_id:02d}_{tile_index:03d}"
            volume = stage.DefinePrim(child_path, "NavMeshVolume")
            volume.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set(
                [Gf.Vec3f(-0.5), Gf.Vec3f(0.5)]
            )
            xform = UsdGeom.Xformable(volume)
            xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d((left + right) * 0.5, (bottom + top) * 0.5, z + 1.0)
            )
            xform.AddScaleOp().Set(Gf.Vec3f(right - left, top - bottom, 4.0))
            volume_records.append(
                {
                    "path": child_path,
                    "component_id": int(component_id),
                    "bounds_xyxy": [left, bottom, right, top],
                }
            )
    return {
        "mesh_path": mesh_path,
        "volume_root_path": volume_path,
        "volume_count": len(volume_records),
        "volumes": volume_records,
        "quad_count": len(counts),
        "source_scene_path": source_scene_path,
        "exclusion_strategy": exclusion_strategy,
        "excluded_top_level_paths": excluded_top_level_paths,
        "excluded_geometry_count": len(excluded_geometry_paths),
        "excluded_geometry_path_samples": excluded_geometry_paths[:32],
        "hidden_scene_branch_count": len(hidden_scene_branches),
        "hidden_scene_branches": hidden_scene_branches,
        "inclusion_policy": (
            "default include; every pre-existing /World branch hidden during bake"
            if exclusion_strategy == "temporary_visibility"
            else "default include; every pre-existing /World branch excluded by API"
        ),
        "render_visibility_during_bake": (
            "approved helper visible; original scene branches invisible"
            if exclusion_strategy == "temporary_visibility"
            else "inherited"
        ),
        "render_visibility_after_bake": "invisible",
        "ground_z_m": z,
        "component_count": len(regions.component_ids),
    }


def finalize_approved_navmesh_surface(
    stage: Any,
    *,
    mesh_path: str = "/World/ApprovedWalkingSurface",
    volume_path: str = "/World/ApprovedWalkingNavMeshVolumes",
    scene_visibility_records: Sequence[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Hide and de-collide the helper after Recast has copied its polygons.

    ``scene_visibility_records`` are the ``hidden_scene_branches`` returned by
    :func:`author_approved_navmesh_surface` under the ``temporary_visibility``
    strategy; when provided they are restored so the source scene reappears.
    """

    from pxr import UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        raise RuntimeError(f"approved NavMesh helper is missing: {mesh_path}")
    UsdGeom.Imageable(prim).MakeInvisible()
    volume_prim = stage.GetPrimAtPath(volume_path)
    if volume_prim.IsValid():
        UsdGeom.Imageable(volume_prim).MakeInvisible()
    collision_removed = bool(prim.RemoveAPI(UsdPhysics.CollisionAPI))
    restored_branches = (
        restore_scene_visibility(stage, scene_visibility_records)
        if scene_visibility_records
        else 0
    )
    return {
        "mesh_path": mesh_path,
        "volume_root_path": volume_path,
        "render_visibility": "invisible",
        "volume_render_visibility": (
            "invisible" if volume_prim.IsValid() else "missing"
        ),
        "collision_api_removed": collision_removed,
        "restored_scene_branch_count": restored_branches,
    }
