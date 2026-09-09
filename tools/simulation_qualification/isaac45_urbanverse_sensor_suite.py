#!/usr/bin/env python3
"""Three-camera sensor evidence for RTX 3090 + Isaac Sim 4.5.

One invocation opens one run-local wrapper USD and captures the supplied
centre/left/right calibration in one Replicator step. Isaac Sim 4.5 only
provides native ``pinhole`` and fisheye-polynomial models here, so the requested
OpenCV K/D values are preserved as provenance and their non-equivalence is
reported explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from appearance_randomization import isaac45_urbanverse_appearance_randomization_smoke as base  # noqa: E402
from urbanverse.viz_style import load_font  # noqa: E402


# Keep the requested vehicle rig local to this Isaac 4.5 script. Importing the
# Isaac 6 probe here can initialise Kit modules before SimulationApp and also
# incorrectly implies that its newer camera API is available in 4.5.
CAMERA_CALIBRATIONS: list[dict[str, Any]] = [
    {
        "id": "camera_tm_pinhole",
        "topic": "/camera_tm_pinhole/image_color/compressed",
        "requested_projection_type": "pinholeOpenCV",
        "native_projection_type": "pinhole",
        "source_document_model": "fisheye",
        "source_label_ambiguity": "The source block says model=fisheye while the topic/id says pinhole; this test follows the pinhole topic/id and flags the conflict.",
        "role": "center_forward_pinhole",
        "position_vehicle_xyz_m": [0.235, 0.0, 0.22],
        "rpy_vehicle_camera_rad": [-1.53274814910142, 0.0340339204138882, -1.628915790886308],
        "K": [1015.185477, 0.0, 959.393487, 0.0, 1015.185477, 770.438403, 0.0, 0.0, 1.0],
        "D": [-0.06166825, -0.01122689, 0.00379161, -0.00141415],
        "image_size": [1920, 1536],
    },
    {
        "id": "camera_tm1_fisheye",
        "topic": "/camera_tm1_fisheye/image_color/compressed",
        "requested_projection_type": "fisheyeOpenCV",
        "native_projection_type": "fisheyePolynomial",
        "source_document_model": "fisheye",
        "role": "left_fisheye",
        "position_vehicle_xyz_m": [-0.29, 0.12, 0.18],
        "rpy_vehicle_camera_rad": [0.0, 0.0, -1.57],
        "K": [444.868696406, 0.0, 959.74232288, 0.0, 444.914755474, 767.235175875, 0.0, 0.0, 1.0],
        "D": [0.154742964974, -0.0742392825496, 0.0229376011173, -0.00285936600559],
        "image_size": [1920, 1536],
    },
    {
        "id": "camera_tm2_fisheye",
        "topic": "/camera_tm2_fisheye/image_color/compressed",
        "requested_projection_type": "fisheyeOpenCV",
        "native_projection_type": "fisheyePolynomial",
        "source_document_model": "fisheye",
        "role": "right_fisheye",
        "position_vehicle_xyz_m": [-0.29, -0.12, 0.18],
        "rpy_vehicle_camera_rad": [0.0, 0.0, 1.57],
        "K": [444.868696406, 0.0, 959.74232288, 0.0, 444.914755474, 767.235175875, 0.0, 0.0, 1.0],
        "D": [0.154742964974, -0.0742392825496, 0.0229376011173, -0.00285936600559],
        "image_size": [1920, 1536],
    },
]


def camera_intrinsics(cal: dict[str, Any], sensor_width_mm: float, width: int, height: int) -> dict[str, float]:
    k = cal["K"]
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    sensor_width_mm = float(sensor_width_mm)
    sensor_height_mm = sensor_width_mm * float(height) / float(width)
    focal_x_mm = fx * sensor_width_mm / float(width)
    focal_y_mm = fy * sensor_height_mm / float(height)
    return {
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
        "focal_length_mm": 0.5 * (focal_x_mm + focal_y_mm),
        "horizontal_aperture_mm": sensor_width_mm,
        "vertical_aperture_mm": sensor_height_mm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("craftbench", "training"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--usd-sha256")
    parser.add_argument("--tar-sha256")
    parser.add_argument("--camera-path", default="auto")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--wrapper-template", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--settle-frames", type=int, default=24)
    parser.add_argument("--rt-subframes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=base.json_default) + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_rgb(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise RuntimeError(f"unexpected RGB shape {array.shape}")
    return np.clip(array[..., :3], 0, 255).astype(np.uint8)


def normalise_depth(value: Any) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    return np.asarray(value, dtype=np.float32).squeeze()


def vec3(value) -> np.ndarray:
    return np.asarray([float(value[i]) for i in range(3)], dtype=np.float64)


def calibration_for_resolution(cal: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Scale the supplied 1920x1536 K matrix without changing fisheye D."""
    result = dict(cal)
    k = [float(value) for value in cal["K"]]
    source_width, source_height = [float(value) for value in cal["image_size"]]
    sx, sy = float(width) / source_width, float(height) / source_height
    k[0] *= sx
    k[2] *= sx
    k[4] *= sy
    k[5] *= sy
    result["K"] = k
    result["image_size"] = [int(width), int(height)]
    result["source_image_size"] = list(cal["image_size"])
    return result


def camera_matrices(base_matrix):
    """Author a reproducible visual rig around the selected centre view.

    The centre camera uses the selected scene camera exactly.  Side-camera
    translations use the supplied camera-to-vehicle offsets, while their
    visual directions are yawed left/right because the documented side-camera
    RPY lacks an unambiguous ROS optical-frame mapping.
    """
    from pxr import Gf

    centre_cal = CAMERA_CALIBRATIONS[0]
    centre_eye = base_matrix.ExtractTranslation()
    forward = base_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    right = base_matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized()
    left = -right
    up = base_matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetNormalized()
    rows: dict[str, Any] = {}
    for cal in CAMERA_CALIBRATIONS:
        delta = np.asarray(cal["position_vehicle_xyz_m"], dtype=np.float64) - np.asarray(
            centre_cal["position_vehicle_xyz_m"], dtype=np.float64
        )
        eye = centre_eye + forward * float(delta[0]) + left * float(delta[1]) + up * float(delta[2])
        matrix = Gf.Matrix4d(base_matrix)
        if cal["role"] == "left_fisheye":
            side_forward = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(up, 90.0)).TransformDir(forward).GetNormalized()
            matrix = Gf.Matrix4d(1.0).SetLookAt(eye, eye + side_forward, up).GetInverse()
        elif cal["role"] == "right_fisheye":
            side_forward = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(up, -90.0)).TransformDir(forward).GetNormalized()
            matrix = Gf.Matrix4d(1.0).SetLookAt(eye, eye + side_forward, up).GetInverse()
        matrix.SetTranslateOnly(eye)
        rows[cal["id"]] = matrix
    return rows


def readback_metrics(camera_params: Any, cal: dict[str, Any], camera_prim: Any) -> dict[str, Any]:
    if not isinstance(camera_params, dict):
        return {"status": "missing", "raw_type": type(camera_params).__name__}
    requested_k = cal["K"]
    requested = {
        "fx": float(requested_k[0]),
        "fy": float(requested_k[4]),
        "cx": float(requested_k[2]),
        "cy": float(requested_k[5]),
    }
    centre = camera_params.get("cameraFisheyeOpticalCentre")
    try:
        centre_values = np.asarray(centre).reshape(-1) if centre is not None else np.empty((0,))
    except Exception:
        centre_values = np.empty((0,))
    actual_from_annotator = {
        "fx": camera_params.get("cameraOpenCVFx"),
        "fy": camera_params.get("cameraOpenCVFy"),
        "cx": centre_values[0] if centre_values.size >= 2 else None,
        "cy": centre_values[1] if centre_values.size >= 2 else None,
    }
    authored = {}
    for name in (
        "cameraProjectionType",
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "fthetaWidth",
        "fthetaHeight",
        "fthetaCx",
        "fthetaCy",
        "fthetaMaxFov",
        "fthetaPolyA",
        "fthetaPolyB",
    ):
        attr = camera_prim.GetAttribute(name) if camera_prim and camera_prim.IsValid() else None
        authored[name] = base.json_default(attr.Get()) if attr and attr.IsValid() else None
    derived_usd_k = None
    try:
        width, height = [float(value) for value in cal["image_size"]]
        focal = float(authored["focalLength"])
        horizontal_aperture = float(authored["horizontalAperture"])
        vertical_aperture = float(authored["verticalAperture"])
        horizontal_offset = float(authored["horizontalApertureOffset"] or 0.0)
        vertical_offset = float(authored["verticalApertureOffset"] or 0.0)
        derived_usd_k = {
            "fx": focal / horizontal_aperture * width,
            "fy": focal / vertical_aperture * height,
            "cx": width * (0.5 + horizontal_offset / horizontal_aperture),
            "cy": height * (0.5 - vertical_offset / vertical_aperture),
        }
    except Exception:
        pass
    errors = {}
    for key in requested:
        try:
            errors[key + "_absolute_px"] = abs(float(actual_from_annotator[key]) - requested[key])
        except Exception:
            errors[key + "_absolute_px"] = None
    derived_errors = {}
    for key in requested:
        try:
            derived_errors[key + "_absolute_px"] = abs(float(derived_usd_k[key]) - requested[key])
        except Exception:
            derived_errors[key + "_absolute_px"] = None
    return {
        "camera_model": camera_params.get("cameraModel"),
        "native_projection_authored": authored.get("cameraProjectionType"),
        "requested": requested,
        "annotator_actual": actual_from_annotator,
        "absolute_errors_px": errors,
        "authored_attributes": authored,
        "derived_usd_pinhole_K": derived_usd_k,
        "derived_usd_K_absolute_errors_px": derived_errors,
        "fisheye_nominal_width": camera_params.get("cameraFisheyeNominalWidth"),
        "fisheye_nominal_height": camera_params.get("cameraFisheyeNominalHeight"),
        "fisheye_max_fov": camera_params.get("cameraFisheyeMaxFOV"),
    }


def make_contact_sheet(records: list[dict[str, Any]], output: Path, scene: str) -> None:
    cell_w, cell_h, header = 540, 500, 72
    sheet = Image.new("RGB", (cell_w * 3, header + cell_h), (242, 245, 247))
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(24, bold=True)
    body_font = load_font(15)
    draw.text((18, 16), f"RTX 3090 · Isaac 4.5 三相机同步证据 · {scene}", fill=(18, 25, 31), font=title_font)
    for index, record in enumerate(records):
        path = Path(record["rgb"])
        # Avoid Pillow's context-manager path inside Kit: Isaac's bundled
        # ImageFile module can be older than the venv Pillow Image module.
        image = Image.open(path).convert("RGB")
        image.thumbnail((520, 416), Image.Resampling.LANCZOS)
        x, y = index * cell_w + 10, header + 8
        sheet.paste(image, (x, y))
        model = record["camera_readback"].get("camera_model")
        draw.text((x, y + 424), f"{record['camera_id']} · {model}", fill=(22, 31, 37), font=body_font)
        draw.text(
            (x, y + 449),
            f"finite depth={record['metrics'].get('finite_depth_ratio', 0.0):.3f}",
            fill=(45, 57, 65),
            font=body_font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def make_response_sheet(records: list[dict[str, Any]], output: Path, scene: str) -> None:
    cell_w, cell_h, header = 480, 360, 70
    sheet = Image.new("RGB", (cell_w * len(records), header + cell_h), (242, 245, 247))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 16), f"曝光响应与噪声边界 · {scene}", fill=(18, 25, 31), font=load_font(24, bold=True))
    for index, record in enumerate(records):
        image = Image.open(record["rgb"]).convert("RGB")
        image.thumbnail((460, 290), Image.Resampling.LANCZOS)
        x, y = index * cell_w + 10, header + 8
        sheet.paste(image, (x, y))
        draw.text((x, y + 298), record["label"], fill=(22, 31, 37), font=load_font(16, bold=True))
        draw.text(
            (x, y + 325),
            f"MAE={record.get('mae_from_baseline_255', 0.0):.3f}/255",
            fill=(45, 57, 65),
            font=load_font(14),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    captures_dir = run_dir / "captures"
    metadata_dir = run_dir / "metadata"
    vis_dir = run_dir / "visualizations"
    wrapper_dir = run_dir / "wrapper"
    for directory in (captures_dir, metadata_dir, vis_dir, wrapper_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_usd = args.usd.resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(source_usd)
    wrapper_path = wrapper_dir / f"{args.scene}_sensor_wrapper.usda"
    wrapper_path.write_text(
        args.wrapper_template.read_text(encoding="utf-8").replace("SOURCE_USD", str(source_usd)), encoding="utf-8"
    )
    summary_path = metadata_dir / "summary.json"
    summary: dict[str, Any] = {
        "status": "failed",
        "test_point": 3,
        "family": args.family,
        "scene": args.scene,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_usd": str(source_usd),
        "wrapper_usd": str(wrapper_path),
    }
    write_json(summary_path, summary)
    app = None
    try:
        from isaacsim import SimulationApp

        launch = {
            "headless": True,
            "renderer": "RayTracedLighting",
            "width": args.width,
            "height": args.height,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "multi_gpu": False,
        }
        print("SIMULATION_APP " + json.dumps(launch, sort_keys=True), flush=True)
        app = SimulationApp(launch, experience=str(args.experience.resolve()))
        import carb
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdGeom

        settings = carb.settings.get_settings()
        settings.set("/rtx/multiGpu/enabled", False)
        settings.set("/physics/cudaDevice", args.gpu)
        settings.set("/renderer/activeGpu", args.gpu)
        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(wrapper_path)):
            raise RuntimeError("open_stage returned false")
        for update in range(2400):
            app.update()
            try:
                status = tuple(ctx.get_stage_loading_status())
                if not status or (len(status) >= 3 and (int(status[-1]) == 0 or int(status[-2]) >= int(status[-1]))):
                    break
            except Exception:
                if update > 10:
                    break
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError("stage is None")
        audit = base.stage_audit(stage, source_usd.parent)
        stage.SetEditTarget(stage.GetRootLayer())
        stage.DefinePrim("/UrbanVerseSensorEvaluation", "Scope")
        selected, _, base_matrix, selected_meta = base.choose_camera(
            stage, args.camera_path, audit["bounds"], audit["up_axis"], audit["meters_per_unit"]
        )
        selected.GetPrim().SetActive(False)
        matrices = camera_matrices(base_matrix)
        render_products = []
        camera_rows = []
        annotators = []
        semantic_errors = []
        for cal in CAMERA_CALIBRATIONS:
            render_cal = calibration_for_resolution(cal, args.width, args.height)
            intrinsics = camera_intrinsics(render_cal, 20.955, args.width, args.height)
            horizontal_offset = (
                (float(intrinsics["cx_px"]) - args.width * 0.5)
                / float(args.width)
                * float(intrinsics["horizontal_aperture_mm"])
            )
            vertical_offset = (
                (args.height * 0.5 - float(intrinsics["cy_px"]))
                / float(args.height)
                * float(intrinsics["vertical_aperture_mm"])
            )
            kwargs = {
                "parent": "/UrbanVerseSensorEvaluation",
                "name": cal["id"],
                "projection_type": cal["native_projection_type"],
                "focal_length": float(intrinsics["focal_length_mm"]),
                "horizontal_aperture": float(intrinsics["horizontal_aperture_mm"]),
                "horizontal_aperture_offset": horizontal_offset,
                "vertical_aperture_offset": vertical_offset,
                "clipping_range": (0.02, 1000.0),
                "fisheye_nominal_width": float(args.width),
                "fisheye_nominal_height": float(args.height),
                "fisheye_optical_centre_x": float(intrinsics["cx_px"]),
                "fisheye_optical_centre_y": float(intrinsics["cy_px"]),
                "fisheye_max_fov": 190.0,
                "fisheye_polynomial_a": 0.0,
                "fisheye_polynomial_b": 1.0 / max(float(intrinsics["fx_px"]), 1.0),
                "focus_distance": 10.0,
                "f_stop": 0.0,
            }
            camera_item = rep.create.camera(**kwargs)
            output_prims = camera_item.get_output_prims().get("prims", [])
            if not output_prims:
                raise RuntimeError(f"Replicator did not return a prim for {cal['id']}")
            camera_xform = output_prims[0]
            camera_children = [child for child in camera_xform.GetChildren() if child.IsA(UsdGeom.Camera)]
            if len(camera_children) != 1:
                raise RuntimeError(
                    f"expected one Camera below {camera_xform.GetPath()}, found {len(camera_children)}"
                )
            camera_prim = camera_children[0]
            xform = UsdGeom.Xformable(camera_xform)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(matrices[cal["id"]])
            # The selected source-camera matrix already contains the complete
            # world orientation. Remove Replicator's automatic Z-up camera
            # child rotation so it is not applied a second time.
            UsdGeom.Xformable(camera_prim).ClearXformOpOrder()
            UsdGeom.Camera(camera_prim).GetVerticalApertureAttr().Set(
                float(intrinsics["vertical_aperture_mm"])
            )
            render_product = rep.create.render_product(camera_item, (args.width, args.height), force_new=True)
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            params = rep.AnnotatorRegistry.get_annotator("camera_params")
            for annotator in (rgb, depth, params):
                annotator.attach([render_product])
            semantic = None
            semantic_error = None
            try:
                semantic = rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"colorize": True}
                )
                semantic.attach([render_product])
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
                semantic = None
            render_products.append(render_product)
            annotators.append((rgb, depth, params, semantic))
            semantic_errors.append(semantic_error)
            camera_rows.append(
                {
                    "calibration": cal,
                    "render_calibration": render_cal,
                    "intrinsics_used": intrinsics,
                    "xform_path": str(camera_xform.GetPath()),
                    "prim_path": str(camera_prim.GetPath()),
                    "camera_prim": camera_prim,
                }
            )
        stage.GetRootLayer().Save()
        for _ in range(args.settle_frames):
            app.update()
        timeline = omni.timeline.get_timeline_interface()
        time_before = float(timeline.get_current_time())
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        try:
            rep.orchestrator.wait_until_complete()
        except Exception as exc:
            print(f"WAIT_NOTE {exc!r}", flush=True)
        time_after = float(timeline.get_current_time())
        records = []
        captured_rgb: dict[str, np.ndarray] = {}
        for cal, row, trio, semantic_error in zip(CAMERA_CALIBRATIONS, camera_rows, annotators, semantic_errors):
            rgb_array = normalise_rgb(trio[0].get_data())
            depth_array = normalise_depth(trio[1].get_data())
            camera_params = trio[2].get_data()
            capture = captures_dir / cal["id"]
            capture.mkdir(parents=True, exist_ok=True)
            rgb_path = capture / "rgb.png"
            Image.fromarray(rgb_array, mode="RGB").save(rgb_path)
            captured_rgb[cal["id"]] = rgb_array
            base.save_depth(depth_array, capture / "distance_to_camera.npy", capture / "distance_to_camera_vis.png")
            write_json(capture / "camera_params.json", camera_params)
            semantic_path = None
            semantic_info_path = None
            semantic_shape = None
            semantic_unique_colors = None
            if trio[3] is not None:
                semantic_value = trio[3].get_data()
                semantic_data = semantic_value.get("data") if isinstance(semantic_value, dict) else semantic_value
                semantic_array = np.asarray(semantic_data)
                semantic_shape = list(semantic_array.shape)
                if semantic_array.ndim == 2 and semantic_array.dtype.itemsize == 4:
                    # Isaac 4.5 commonly returns colorized semantics as packed
                    # uint32 RGBA rather than an HxWx4 uint8 array.
                    semantic_array = np.ascontiguousarray(semantic_array).view(np.uint8).reshape(
                        semantic_array.shape[0], semantic_array.shape[1], 4
                    )
                if semantic_array.ndim == 3 and semantic_array.shape[-1] >= 3:
                    semantic_rgb = np.clip(semantic_array[..., :3], 0, 255).astype(np.uint8)
                    semantic_path = capture / "semantic_segmentation.png"
                    Image.fromarray(semantic_rgb, mode="RGB").save(semantic_path)
                    semantic_unique_colors = int(
                        np.unique(semantic_rgb.reshape(-1, 3), axis=0).shape[0]
                    )
                if isinstance(semantic_value, dict):
                    semantic_info_path = capture / "semantic_info.json"
                    write_json(
                        semantic_info_path,
                        {key: value for key, value in semantic_value.items() if key != "data"},
                    )
            metrics = base.image_metrics(rgb_array, depth_array)
            readback = readback_metrics(camera_params, row["render_calibration"], row["camera_prim"])
            actual_matrix = UsdGeom.Xformable(row["camera_prim"]).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            expected_translation = matrices[cal["id"]].ExtractTranslation()
            actual_translation = actual_matrix.ExtractTranslation()
            translation_error = float(
                np.linalg.norm(
                    np.asarray([float(actual_translation[i] - expected_translation[i]) for i in range(3)])
                )
            )
            expected_model = cal["native_projection_type"]
            record = {
                "camera_id": cal["id"],
                "role": cal["role"],
                "topic": cal["topic"],
                "rgb": str(rgb_path),
                "distance": str(capture / "distance_to_camera.npy"),
                "distance_visualization": str(capture / "distance_to_camera_vis.png"),
                "camera_params": str(capture / "camera_params.json"),
                "semantic_segmentation": str(semantic_path) if semantic_path else None,
                "semantic_info": str(semantic_info_path) if semantic_info_path else None,
                "semantic_shape": semantic_shape,
                "semantic_unique_color_count": semantic_unique_colors,
                "semantic_error": semantic_error,
                "metrics": metrics,
                "camera_readback": readback,
                "requested_calibration_model": cal["requested_projection_type"],
                "expected_native_camera_model": expected_model,
                "native_projection_model_confirmed": readback.get("native_projection_authored") == expected_model,
                "opencv_D": cal["D"],
                "opencv_D_exact_visual_equivalence": False,
                "opencv_D_status": "not satisfied: four-coefficient OpenCV equivalence is not represented by this polynomial smoke",
                "world_transform": base.matrix_rows(matrices[cal["id"]]),
                "actual_world_transform": base.matrix_rows(actual_matrix),
                "authored_translation_error_stage_units": translation_error,
                **{key: value for key, value in row.items() if key != "camera_prim"},
            }
            records.append(record)
        centre_id = CAMERA_CALIBRATIONS[0]["id"]
        centre_baseline = captured_rgb[centre_id]
        exposure_records = []
        for label, exposure_ev in (("0 EV 重复帧", 0.0), ("-2 EV", -2.0), ("+2 EV", 2.0)):
            settings.set("/rtx/post/tonemap/exposure", exposure_ev)
            for _ in range(3):
                app.update()
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
            try:
                rep.orchestrator.wait_until_complete()
            except Exception:
                pass
            exposure_rgb = normalise_rgb(annotators[0][0].get_data())
            exposure_path = captures_dir / "exposure_response" / f"exposure_{exposure_ev:+.1f}_ev.png"
            exposure_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(exposure_rgb, mode="RGB").save(exposure_path)
            delta = exposure_rgb.astype(np.float32) - centre_baseline.astype(np.float32)
            exposure_records.append(
                {
                    "label": label,
                    "requested_exposure_ev": exposure_ev,
                    "readback_exposure_ev": settings.get("/rtx/post/tonemap/exposure"),
                    "rgb": str(exposure_path),
                    "mae_from_baseline_255": float(np.abs(delta).mean()),
                    "absolute_mean_luma_delta_255": float(
                        abs(exposure_rgb.astype(np.float32).mean() - centre_baseline.astype(np.float32).mean())
                    ),
                }
            )
        settings.set("/rtx/post/tonemap/exposure", 0.0)
        rng = np.random.default_rng(args.seed)
        sigma = 6.0
        noise = rng.normal(0.0, sigma, size=centre_baseline.shape)
        noisy_rgb = np.clip(centre_baseline.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noisy_path = captures_dir / "sensor_noise_proxy" / "gaussian_read_noise_proxy.png"
        noisy_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(noisy_rgb, mode="RGB").save(noisy_path)
        noise_record = {
            "label": "高斯读出噪声 proxy",
            "rgb": str(noisy_path),
            "mae_from_baseline_255": float(
                np.abs(noisy_rgb.astype(np.float32) - centre_baseline.astype(np.float32)).mean()
            ),
            "sigma_255": sigma,
            "seed": args.seed,
            "native_isaac_sensor_noise": False,
            "disclosure": "Deterministic post-render Gaussian noise proxy; no measured physical sensor response model was supplied.",
        }
        response_sheet_records = [*exposure_records, noise_record]
        response_sheet = vis_dir / "exposure_noise_contact_sheet.png"
        make_response_sheet(response_sheet_records, response_sheet, args.scene)
        for trio in annotators:
            for annotator in trio:
                try:
                    annotator.detach()
                except Exception:
                    pass
        for render_product in render_products:
            try:
                render_product.destroy()
            except Exception:
                pass
        contact_sheet = vis_dir / "contact_sheet.png"
        make_contact_sheet(records, contact_sheet, args.scene)
        kit_version = None
        try:
            kit_version = omni.kit.app.get_app().get_kit_version()
        except Exception:
            kit_version = settings.get("/app/kitVersion") or settings.get("/app/buildVersion")
        environment = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": platform.node(),
            "gpu_index": args.gpu,
            "gpu": run_text(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=index,name,memory.used,utilization.gpu,driver_version", "--format=csv,noheader"]),
            "isaac_sim_version": package_version("isaacsim"),
            "replicator_version": package_version("isaacsim-replicator"),
            "isaac_lab_release": (PROJECT_ROOT / "repos" / "IsaacLab" / "VERSION").read_text(encoding="utf-8").strip(),
            "isaac_lab_python_distribution_version": package_version("isaaclab"),
            "torch_version": package_version("torch"),
            "kit_version": base.json_default(kit_version),
            "python_version": sys.version.replace("\n", " "),
            "git_commit": run_text(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]),
            "command_line": [sys.executable, *sys.argv],
            "renderer": "RayTracedLighting",
            "resolution": [args.width, args.height],
            "source_usd": str(source_usd),
            "source_usd_sha256": args.usd_sha256 or sha256(source_usd),
            "source_tar": str(args.tar.resolve()) if args.tar else None,
            "source_tar_sha256": (args.tar_sha256 or sha256(args.tar.resolve())) if args.tar else None,
            "wrapper_path": str(wrapper_path),
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "random_seed": args.seed,
            "camera_selection": selected_meta,
            "spatial_unit_policy": "Supplied rig offsets in metres are applied 1:1 to stage numeric coordinates because the seven scenes were visually validated with metre-like authored coordinates; source metersPerUnit=0.01 remains an explicit metadata inconsistency.",
        }
        write_json(metadata_dir / "environment.json", environment)
        write_json(
            metadata_dir / "camera.json",
            {
                "selected_view": selected_meta,
                "rig": [
                    {key: value for key, value in row.items() if key != "camera_prim"}
                    for row in camera_rows
                ],
            },
        )
        write_json(metadata_dir / "records.json", {"records": records})
        write_json(metadata_dir / "exposure_noise_response.json", {"exposure": exposure_records, "noise": noise_record})
        repeat_luma_delta = exposure_records[0]["absolute_mean_luma_delta_255"]
        exposure_threshold = max(2.0, repeat_luma_delta * 3.0)
        exposure_response_verified = all(
            row["absolute_mean_luma_delta_255"] > exposure_threshold for row in exposure_records[1:]
        )
        pinhole_records = [record for record in records if record["expected_native_camera_model"] == "pinhole"]
        checks = {
            "three_nonempty_rgb": len(records) == 3 and all(r["metrics"]["nonempty_rgb"] for r in records),
            "three_depth_outputs": len(records) == 3
            and all(r["metrics"].get("finite_depth_ratio") is not None for r in records),
            "three_semantic_outputs": len(records) == 3
            and all(r["semantic_segmentation"] is not None for r in records),
            "semantic_label_diversity_observed": len(records) == 3
            and all((r["semantic_unique_color_count"] or 0) > 1 for r in records),
            "native_projection_models_confirmed": len(records) == 3
            and all(r["native_projection_model_confirmed"] for r in records),
            "pinhole_K_authored_equivalence": len(pinhole_records) == 1
            and all(
                value is not None and float(value) <= 1e-3
                for value in pinhole_records[0]["camera_readback"]["derived_usd_K_absolute_errors_px"].values()
            ),
            "opencv_D_exact_visual_equivalence": False,
            "supplied_translation_extrinsics_applied": len(records) == 3
            and all(r["authored_translation_error_stage_units"] <= 1e-6 for r in records),
            "supplied_rpy_extrinsics_exactly_applied": False,
            "multi_camera_sync_capture": len(records) == 3,
            "exposure_setting_readback": all(
                abs(float(row["readback_exposure_ev"]) - row["requested_exposure_ev"]) < 1e-6
                for row in exposure_records
            ),
            "exposure_visual_response_verified": exposure_response_verified,
            "native_physical_noise_response": False,
        }
        execution_ok = all(
            checks[key]
            for key in (
                "three_nonempty_rgb",
                "three_depth_outputs",
                "three_semantic_outputs",
                "native_projection_models_confirmed",
                "pinhole_K_authored_equivalence",
                "supplied_translation_extrinsics_applied",
                "multi_camera_sync_capture",
            )
        )
        summary.update(
            {
                "status": "success" if execution_ok else "failed",
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_s": round(time.perf_counter() - started, 3),
                "audit": audit,
                "environment": environment,
                "records": records,
                "same_replicator_step": True,
                "timeline_time_before": time_before,
                "timeline_time_after": time_after,
                "checks": checks,
                "failure_stage": None if execution_ok else "sensor_execution_validation",
                "error": None if execution_ok else "one or more required RGB/depth/semantic/model/K/extrinsic artifacts failed validation",
                "status_assessment": "partial: synchronized RGB/depth and native camera models tested; exact OpenCV D and physical noise response not satisfied",
                "limitations": [
                    "Side-camera directions use a recorded visual left/right mapping because the supplied RPY optical-frame convention is ambiguous.",
                    "OpenCV fisheye D is recorded but not claimed equivalent to Isaac's fisheye polynomial interface.",
                    "The centre calibration block labels its model as fisheye while its topic/id says pinhole; this run follows the pinhole topic/id, so the authoritative model still needs user confirmation.",
                    "No measured physical-camera noise or exposure response dataset was supplied.",
                    "The saved Gaussian noise image is a deterministic post-render proxy, not native sensor-noise validation.",
                    "A semantic channel can be emitted, but source label completeness/accuracy is not established by colorized output alone.",
                    "Rig translations assume one authored stage coordinate equals one practical metre despite source metersPerUnit=0.01 metadata.",
                ],
                "exposure_response": {
                    "records": exposure_records,
                    "repeat_luma_delta_255": repeat_luma_delta,
                    "visual_threshold_255": exposure_threshold,
                    "verified": exposure_response_verified,
                },
                "noise_response": noise_record,
                "visualizations": {"contact_sheet": str(contact_sheet), "exposure_noise": str(response_sheet)},
            }
        )
        write_json(summary_path, summary)
        return 0 if execution_ok else 1
    except Exception as exc:
        traceback.print_exc()
        summary.update(
            {
                "status": "failed",
                "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_s": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(summary_path, summary)
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
