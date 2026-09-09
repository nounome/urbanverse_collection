#!/usr/bin/env python3
"""UrbanVerse probes for constant, replayed, and closed-loop Go2 commands."""

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
from PIL import Image, ImageDraw, ImageEnhance


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from urbanverse.dynamic_agents.navigation import control as navigation  # noqa: E402
from urbanverse.dynamic_agents.integration import go2_route_capture as capture  # noqa: E402
from urbanverse.dynamic_agents.config import TrafficSceneConfig  # noqa: E402
from urbanverse.dynamic_agents.integration.isaac45_go2z1_v2_compat_probe import (  # noqa: E402
    adapt_pure_go2_observation,
    build_actor,
)
from urbanverse.dynamic_agents.traffic import (  # noqa: E402
    ContinuousVehicleManager,
    MultiVehicleManager,
)
from urbanverse.dynamic_agents.integration.scene_setup import (  # noqa: E402
    author_portable_traffic_visuals,
    configure_go2_support_corridor,
    configure_preauthored_traffic_visuals,
)
from urbanverse.dynamic_agents.pedestrians import (  # noqa: E402
    OfficialPeopleManager,
    ProjectPeopleRoamingManager,
    OfficialPeopleRoamingManager,
    OfficialPeopleRuntime,
    WalkableRegions,
    align_initial_targets_to_waypoint_loops,
    assignments_to_specs,
    author_approved_navmesh_surface,
    build_roaming_waypoint_loops,
    expand_waypoint_loops_with_grid_paths,
    finalize_approved_navmesh_surface,
    author_official_people,
    author_people_payload,
    build_project_people_payload,
    configure_preauthored_people,
    enable_official_people_runtime,
    plan_roaming_assignments,
)
from urbanverse.dynamic_agents.micromobility import (  # noqa: E402
    MixedWalkableAgentAudit,
    MicromobilityAgentSpec,
    MicromobilityRoamingManager,
    MicromobilityUsdRuntime,
    convert_assets as convert_micromobility_assets,
    balanced_component_assignment,
    eligible_micromobility_components,
    load_catalog as load_micromobility_catalog,
)
from urbanverse.dynamic_agents.collision_passthrough.config import (  # noqa: E402
    load_experiment_config,
)
from urbanverse.dynamic_agents.collision_passthrough.authoring import (  # noqa: E402
    configure_experiment_obstacles,
)
from urbanverse.dynamic_agents.core.mixed_avoidance import (  # noqa: E402
    reciprocal_speed_scales,
)
from urbanverse.dynamic_agents.review.mixed_trajectory import (  # noqa: E402
    build_mixed_actual_trajectory_map,
)
from urbanverse.dynamic_agents.navigation.policy_selection import DEFAULT_POLICY_KIND, EXTERNAL_SOURCE
from urbanverse.viz_style import load_font  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("constant", "replay", "route", "three_point"), required=True)
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--source-tar", type=Path)
    parser.add_argument("--reference-route", type=Path, required=True)
    parser.add_argument("--replay-trajectory", type=Path)
    parser.add_argument("--maneuver-plan", type=Path)
    parser.add_argument("--replay-start-arc", type=float, default=126.3234683161813)
    parser.add_argument("--constant-vx", type=float, default=0.24)
    parser.add_argument("--constant-wz", type=float, default=0.18)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--stop-at-route-goal", action="store_true",
                        help="End mixed capture when Go2 completes its route, rather than wait for duration.")
    parser.add_argument("--lookahead-distance", type=float, default=1.20)
    parser.add_argument("--minimum-tracking-speed", type=float, default=0.24)
    parser.add_argument("--max-forward-speed", type=float, default=0.34)
    parser.add_argument("--max-tracking-yaw-rate", type=float, default=0.18)
    parser.add_argument("--curvature-speed-gain", type=float, default=2.0)
    parser.add_argument("--yaw-command-gain", type=float, default=1.0)
    parser.add_argument("--maximum-cross-track-error", type=float, default=5.0)
    parser.add_argument("--policy-kind", choices=("torchscript", "go2z1_v2_checkpoint", "robot_lab", "himloco"), default=DEFAULT_POLICY_KIND)
    parser.add_argument("--external-policy-source", type=Path, default=EXTERNAL_SOURCE)
    parser.add_argument("--route-review-only", action="store_true")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--wrapper-template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--experience",
        type=Path,
        help=(
            "Override the Kit experience. Official-People runs otherwise use the "
            "maintained compact Isaac Sim 4.5 composition."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--scene10-multivehicle-traffic",
        action="store_true",
        help="Compatibility alias for --multivehicle-traffic.",
    )
    parser.add_argument(
        "--multivehicle-traffic",
        action="store_true",
        help="Run configured kinematic traffic agents beside the articulated Go2.",
    )
    parser.add_argument(
        "--traffic-scene-config",
        type=Path,
        help="Portable scene-specific roads, routes, vehicle catalog, and camera settings.",
    )
    parser.add_argument("--traffic-registry", type=Path)
    parser.add_argument("--traffic-audit-inventory", type=Path)
    parser.add_argument("--traffic-validated-routes", type=Path)
    parser.add_argument("--traffic-validated-static-bodies", type=Path)
    parser.add_argument("--traffic-continuous-looping", action="store_true")
    parser.add_argument("--traffic-automotive-routes", type=Path)
    parser.add_argument("--traffic-vehicle-count", type=int, default=5)
    parser.add_argument("--traffic-initial-fill", action="store_true")
    parser.add_argument(
        "--pedestrian-config",
        type=Path,
        help="Enable animated Isaac Sim People assets using reviewed sidewalk routes.",
    )
    parser.add_argument(
        "--roaming-ghost-config",
        type=Path,
        help=(
            "Enable the Scene-independent official resident-People + micromobility + "
            "Go2 world-passthrough composition described by one maintained config."
        ),
    )
    parser.add_argument(
        "--mixed-roaming-config",
        type=Path,
        help=(
            "Enable project-controlled resident People and micromobility in one "
            "approved walkable union; CharacterManager/NavMesh are not used."
        ),
    )
    parser.add_argument(
        "--roaming-ghost-two-panel-video",
        "--mixed-roaming-two-panel-video",
        dest="roaming_ghost_two_panel_video",
        action="store_true",
        help="Write Go2 oblique-follow and one resident-People overhead-follow panel.",
    )
    parser.add_argument(
        "--roaming-validation-level",
        choices=("smoke", "technical", "formal"),
        default="smoke",
        help="Select duration-appropriate acceptance gates for the roaming composition.",
    )
    parser.add_argument(
        "--disable-road-sweep-video",
        action="store_true",
        help=(
            "Disable only the config-defined road-sweep render product; useful for "
            "bounded fixed-camera RTX isolation probes."
        ),
    )
    parser.add_argument(
        "--disable-roaming-two-panel-video",
        action="store_true",
        help=(
            "Suppress the roaming config's implicit two-panel Replicator capture. "
            "When --overview-video is also set, retain only the native overview camera."
        ),
    )
    parser.add_argument(
        "--ghost-ab-normal",
        action="store_true",
        help=(
            "Ghost A/B normal leg: spawn the same passthrough proxies but keep Go2 "
            "collision enabled so the flowerbed/guard-rail/pickup physically block "
            "the robot (normal-collision baseline for the ghost pass-through contrast)."
        ),
    )
    parser.add_argument(
        "--ghost-ab-headless",
        "--roaming-headless",
        dest="ghost_ab_headless",
        action="store_true",
        help=(
            "Run the roaming composition camera-off: suppress the forced two-panel "
            "video for staged physics, density, and ghost A/B validation. The old "
            "--ghost-ab-headless spelling remains a compatibility alias."
        ),
    )
    parser.add_argument(
        "--pedestrian-lifecycle-validation",
        action="store_true",
        help=(
            "Validate continuous pedestrian spawn/despawn and interaction during a "
            "bounded low-cost route run without requiring Go2 to reach its distant goal."
        ),
    )
    parser.add_argument(
        "--traffic-visibility-diagnostic-convoy",
        action="store_true",
        help="Move all five selected vehicle payloads in a spaced camera-visible convoy.",
    )
    parser.add_argument("--minimum-go2-vehicle-clearance", type=float, default=1.0)
    parser.add_argument("--overview-video", action="store_true")
    parser.add_argument("--go2-three-camera", action="store_true", help="Experimental mixed-run onboard RGB/depth trio; requires short qualification")
    parser.add_argument("--three-camera-width", type=int, default=480)
    parser.add_argument("--three-camera-height", type=int, default=384)
    parser.add_argument("--go2-front-pinhole", action="store_true",
                        help="Single RGB camera attached to Robot/base; no chase-pose writes.")
    parser.add_argument(
        "--joint-three-panel-video",
        action="store_true",
        help=(
            "Write synchronized global, Go2-follow, and pedestrian-follow panels using "
            "the overview camera; requires animated pedestrians."
        ),
    )
    parser.add_argument(
        "--joint-four-panel-video",
        action="store_true",
        help=(
            "Write a synchronized 2x2 video containing a true whole-scene global, "
            "the existing fixed local global, Go2-follow, and pedestrian-follow views."
        ),
    )
    parser.add_argument(
        "--pedestrian-follow-index",
        type=int,
        default=3,
        help="Pedestrian index used by --joint-three-panel-video.",
    )
    parser.add_argument("--overview-width", type=int, default=1280)
    parser.add_argument("--overview-height", type=int, default=720)
    parser.add_argument("--overview-fps", type=float, default=10.0)
    parser.add_argument(
        "--overview-video-codec",
        choices=("vp9", "h264"),
        default="vp9",
        help=(
            "Overview encoder. h264 uses the bundled imageio-ffmpeg binary and is "
            "recommended for long 4K captures; vp9 preserves the historical OpenCV path."
        ),
    )
    parser.add_argument("--overview-h264-crf", type=int, default=14)
    parser.add_argument(
        "--overview-h264-preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
        default="veryfast",
    )
    parser.add_argument(
        "--overview-motion-blur",
        action="store_true",
        help="Enable RTX motion blur for overview RGB; disabled by default for sharp evidence.",
    )
    parser.add_argument("--overview-exposure-ev", type=float, default=0.0)
    parser.add_argument("--overview-output-brightness", type=float, default=1.0)
    parser.add_argument("--overview-focal-length", type=float, default=18.0)
    parser.add_argument(
        "--overview-fixed-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Use a fixed world-space overview camera eye instead of the Go2-relative chase view.",
    )
    parser.add_argument(
        "--overview-fixed-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="World-space look-at target paired with --overview-fixed-eye.",
    )
    parser.add_argument(
        "--overview-motion-start-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Start eye position for an independent constant-speed overview camera path.",
    )
    parser.add_argument(
        "--overview-motion-end-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="End eye position paired with --overview-motion-start-eye.",
    )
    parser.add_argument("--overview-motion-speed-mps", type=float, default=0.55)
    parser.add_argument("--overview-motion-pitch-down-deg", type=float, default=5.0)
    parser.add_argument("--overview-motion-lookahead-m", type=float, default=30.0)
    parser.add_argument(
        "--scene-global-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Whole-scene camera eye used by --joint-four-panel-video.",
    )
    parser.add_argument(
        "--scene-global-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Whole-scene camera target used by --joint-four-panel-video.",
    )
    parser.add_argument("--scene-global-focal-length", type=float, default=18.0)
    parser.add_argument(
        "--overview-preview-time-s",
        type=float,
        help=(
            "Framing-only diagnostic: capture one overview frame at this simulation time, "
            "then stop without requiring route or traffic completion."
        ),
    )
    parser.add_argument(
        "--go2-highlight-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Override the Go2 render material with this linear RGB color; physics is unchanged.",
    )
    parser.add_argument("--go2-highlight-emission", type=float, default=0.10)
    parser.add_argument("--collection-light-scale", type=float, default=0.10)
    parser.add_argument("--collection-dome-light-scale", type=float, default=0.10)
    parser.add_argument("--collection-distant-light-scale", type=float, default=0.10)
    parser.add_argument("--collection-sphere-light-scale", type=float, default=0.10)
    parser.add_argument(
        "--collection-light-path-scale",
        action="append",
        default=[],
        metavar="PRIM_PATH=SCALE",
        help="Exact run-local USD light intensity override; repeat for additional lights.",
    )
    parser.add_argument(
        "--source-dome-background-visible",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Show source DomeLight HDR textures in primary camera rays. When omitted, "
            "traffic-scene overview.source_dome_background_visible is used, otherwise true."
        ),
    )
    parser.add_argument(
        "--source-dome-texture-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use source DomeLight HDR texture files. When omitted, traffic-scene "
            "overview.source_dome_texture_enabled is used, otherwise true."
        ),
    )
    parser.add_argument("--overview-chase-distance", type=float, default=8.0)
    parser.add_argument("--overview-lateral-offset", type=float, default=3.0)
    parser.add_argument("--overview-chase-height", type=float, default=6.5)
    parser.add_argument("--overview-target-forward", type=float, default=3.0)
    parser.add_argument("--overview-target-lateral", type=float, default=0.0)
    parser.add_argument("--minimum-visible-vehicle-pass-duration", type=float, default=3.0)
    parser.add_argument("--opposing-showcase-speed", type=float, default=3.12)
    parser.add_argument(
        "--opposing-showcase-start-route-index",
        type=int,
        default=0,
        help=(
            "Diagnostic-only start index for the opposing showcase vehicle on its "
            "validated route; zero preserves the formal traffic schedule."
        ),
    )
    args = parser.parse_args()
    if (args.overview_fixed_eye is None) != (args.overview_fixed_target is None):
        parser.error("--overview-fixed-eye and --overview-fixed-target must be provided together")
    if (args.overview_motion_start_eye is None) != (
        args.overview_motion_end_eye is None
    ):
        parser.error(
            "--overview-motion-start-eye and --overview-motion-end-eye must be provided together"
        )
    if args.overview_motion_start_eye is not None:
        if args.overview_fixed_eye is not None:
            parser.error("moving and fixed overview camera modes are exclusive")
        if args.overview_motion_speed_mps <= 0.0:
            parser.error("--overview-motion-speed-mps must be positive")
        if not 0.0 <= args.overview_motion_pitch_down_deg < 90.0:
            parser.error("--overview-motion-pitch-down-deg must be in [0, 90)")
        if args.overview_motion_lookahead_m <= 0.0:
            parser.error("--overview-motion-lookahead-m must be positive")
        start_eye = np.asarray(args.overview_motion_start_eye, dtype=np.float64)
        end_eye = np.asarray(args.overview_motion_end_eye, dtype=np.float64)
        if np.linalg.norm(end_eye[:2] - start_eye[:2]) < 1.0e-6:
            parser.error("moving overview camera path must have non-zero XY length")
        args.overview_video = True
    if args.overview_preview_time_s is not None:
        if args.overview_preview_time_s <= 0.0:
            parser.error("--overview-preview-time-s must be positive")
        args.overview_video = True
    if sum(
        bool(value)
        for value in (
            args.joint_three_panel_video,
            args.joint_four_panel_video,
            args.roaming_ghost_two_panel_video,
        )
    ) > 1:
        parser.error("--joint-three-panel-video and --joint-four-panel-video are exclusive")
    if args.roaming_ghost_config is not None and args.mixed_roaming_config is not None:
        parser.error("official legacy roaming and project mixed roaming are exclusive")
    active_roaming_config = args.mixed_roaming_config or args.roaming_ghost_config
    if (
        active_roaming_config is not None
        and not args.ghost_ab_headless
        and not args.disable_roaming_two_panel_video
    ):
        args.roaming_ghost_two_panel_video = True
        args.overview_video = True
    if args.go2_three_camera:
        if args.ghost_ab_headless or args.go2_front_pinhole or args.mixed_roaming_config is None:
            parser.error("three-camera requires camera-enabled project mixed roaming, not front-only/official/headless")
        args.overview_video = True
    if args.go2_front_pinhole:
        args.disable_roaming_two_panel_video = True
        args.disable_road_sweep_video = True
        args.overview_video = True
        args.overview_fixed_eye = args.overview_fixed_target = None
        args.overview_motion_start_eye = args.overview_motion_end_eye = None
        if args.joint_three_panel_video or args.joint_four_panel_video or args.ghost_ab_headless:
            parser.error("front pinhole requires a single camera-enabled view")
    if args.disable_roaming_two_panel_video:
        args.roaming_ghost_two_panel_video = False
    if args.ghost_ab_headless:
        # Camera-off A/B validation: the two-panel video is a formal-run feature,
        # not a physics diagnostic.  Rendering is known to perturb this
        # locomotion environment and trigger non-foot contacts.
        args.roaming_ghost_two_panel_video = False
        args.overview_video = False
    joint_people_video = (
        args.joint_three_panel_video
        or args.joint_four_panel_video
        or args.roaming_ghost_two_panel_video
    )
    if joint_people_video:
        args.overview_video = True
        if args.pedestrian_config is None and active_roaming_config is None:
            parser.error("joint People video capture requires --pedestrian-config")
        if args.pedestrian_follow_index < 0:
            parser.error("--pedestrian-follow-index must be non-negative")
    if (args.scene_global_eye is None) != (args.scene_global_target is None):
        parser.error("--scene-global-eye and --scene-global-target must be provided together")
    if args.joint_four_panel_video and args.scene_global_eye is None:
        parser.error("--joint-four-panel-video requires a scene-global eye and target")
    if args.go2_highlight_color is not None and not all(
        0.0 <= value <= 1.0 for value in args.go2_highlight_color
    ):
        parser.error("--go2-highlight-color components must be in [0, 1]")
    if not 0 <= args.overview_h264_crf <= 51:
        parser.error("--overview-h264-crf must be in [0, 51]")
    if not args.collection_light_path_scale and args.traffic_scene_config is None:
        args.collection_light_path_scale = ["/UrbanVerseAsset/DomeLight_04=1.0"]
    if args.source_dome_background_visible is None:
        configured = True
        if args.traffic_scene_config is not None:
            traffic_payload = json.loads(
                args.traffic_scene_config.resolve().read_text(encoding="utf-8")
            )
            configured = bool(
                traffic_payload.get("overview", {}).get(
                    "source_dome_background_visible", True
                )
            )
        args.source_dome_background_visible = configured
    if args.source_dome_texture_enabled is None:
        configured = True
        if args.traffic_scene_config is not None:
            traffic_payload = json.loads(
                args.traffic_scene_config.resolve().read_text(encoding="utf-8")
            )
            configured = bool(
                traffic_payload.get("overview", {}).get(
                    "source_dome_texture_enabled", True
                )
            )
        args.source_dome_texture_enabled = configured
    return args


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ffmpeg_h264_command(
    executable: str,
    output: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> list[str]:
    """Build the deterministic high-quality 4K overview encoder command."""
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


class FfmpegH264Writer:
    """Small cv2.VideoWriter-compatible wrapper around bundled FFmpeg."""

    def __init__(
        self,
        output: Path,
        fps: float,
        frame_size: tuple[int, int],
        crf: int,
        preset: str,
        log_path: Path,
    ) -> None:
        import imageio_ffmpeg

        self.output = output
        self.width, self.height = frame_size
        self.log_path = log_path
        self._log = log_path.open("wb")
        self.command = ffmpeg_h264_command(
            imageio_ffmpeg.get_ffmpeg_exe(),
            output,
            self.width,
            self.height,
            fps,
            crf,
            preset,
        )
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )
        self._released = False

    def isOpened(self) -> bool:  # noqa: N802 - match OpenCV's API
        return self._process.poll() is None and self._process.stdin is not None

    def write(self, frame: np.ndarray) -> None:
        if self._released or self._process.stdin is None:
            raise RuntimeError("H.264 overview writer is closed")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"unexpected H.264 frame {frame.shape}/{frame.dtype}; "
                f"expected {(self.height, self.width, 3)}/uint8"
            )
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        self._log.close()
        if return_code != 0:
            raise RuntimeError(
                f"H.264 overview encoder exited with {return_code}; see {self.log_path}"
            )


def append_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def pedestrian_follow_camera_view(
    position: np.ndarray,
    heading_rad: float,
    chase_distance: float = 4.0,
    lateral_offset: float = -3.0,
    chase_height: float = 2.2,
    target_forward: float = 1.0,
    target_height: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a close oblique view that keeps a walking person's full body visible."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray([math.cos(heading_rad), math.sin(heading_rad), 0.0])
    left = np.asarray([-forward[1], forward[0], 0.0])
    eye = (
        position
        - forward * chase_distance
        + left * lateral_offset
        + np.asarray([0.0, 0.0, chase_height])
    )
    target = (
        position
        + forward * target_forward
        + np.asarray([0.0, 0.0, target_height])
    )
    return eye, target


def moving_overview_camera_view(
    timestamp_s: float,
    start_eye: np.ndarray,
    end_eye: np.ndarray,
    speed_mps: float,
    pitch_down_deg: float,
    lookahead_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a constant-speed world path with a fixed forward/downward gaze."""
    start_eye = np.asarray(start_eye, dtype=np.float64)
    end_eye = np.asarray(end_eye, dtype=np.float64)
    planar = end_eye[:2] - start_eye[:2]
    path_length = float(np.linalg.norm(planar))
    direction_xy = planar / max(path_length, 1.0e-9)
    travelled = min(path_length, max(0.0, float(timestamp_s)) * float(speed_mps))
    fraction = travelled / max(path_length, 1.0e-9)
    eye = start_eye + fraction * (end_eye - start_eye)
    target = eye.copy()
    target[:2] += direction_xy * float(lookahead_m)
    target[2] -= math.tan(math.radians(float(pitch_down_deg))) * float(
        lookahead_m
    )
    return eye, target


def polyline_camera_view(
    timestamp_s: float,
    eye_points_world_xyz: np.ndarray,
    speed_mps: float,
    pitch_down_deg: float,
    lookahead_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Move a camera at constant speed along a world-space polyline.

    The gaze follows the current planar segment and keeps a constant downward
    pitch.  The final boolean becomes true only once the eye reaches the last
    point; callers can then close an independent video while the simulation and
    other cameras continue.
    """
    points = np.asarray(eye_points_world_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("camera polyline requires at least two XYZ points")
    segments = np.diff(points, axis=0)
    planar_lengths = np.linalg.norm(segments[:, :2], axis=1)
    if np.any(planar_lengths <= 1.0e-9):
        raise ValueError("camera polyline contains a zero-length planar segment")
    cumulative = np.concatenate(([0.0], np.cumsum(planar_lengths)))
    total_length = float(cumulative[-1])
    travelled = min(
        total_length,
        max(0.0, float(timestamp_s)) * float(speed_mps),
    )
    segment_index = min(
        int(np.searchsorted(cumulative, travelled, side="right") - 1),
        len(segments) - 1,
    )
    segment_distance = travelled - float(cumulative[segment_index])
    fraction = segment_distance / float(planar_lengths[segment_index])
    eye = points[segment_index] + fraction * segments[segment_index]
    direction_xy = segments[segment_index, :2] / planar_lengths[segment_index]
    target = eye.copy()
    target[:2] += direction_xy * float(lookahead_m)
    target[2] -= math.tan(math.radians(float(pitch_down_deg))) * float(
        lookahead_m
    )
    reached_endpoint = travelled >= total_length - 1.0e-9
    return eye, target, travelled, total_length, reached_endpoint


def apply_go2_highlight_material(stage: Any, color: list[float], emission: float) -> dict[str, Any]:
    """Add a render-only Go2 highlight without expanding instance proxies.

    Expanding the asset's instance-proxy meshes makes their face-varying UV
    primvars inconsistent in Isaac Sim 4.5 and produces long coloured triangle
    streaks in RTX output.  Keep the source robot untouched and attach one
    small emissive marker to its moving base instead.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    robot_path = "/World/envs/env_0/Robot"
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Go2 prim is missing: {robot_path}")
    material_path = "/UrbanVerseEvaluation/Go2HighlightMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    emissive = [min(1.0, max(0.0, component * emission)) for component in color]
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    marker_path = f"{robot_path}/base/Go2HighlightMarker"
    marker = UsdGeom.Sphere.Define(stage, marker_path)
    marker.CreateRadiusAttr(0.16)
    marker.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.36))
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material)
    resolved, _ = UsdShade.MaterialBindingAPI(marker.GetPrim()).ComputeBoundMaterial()
    if not resolved or str(resolved.GetPath()) != material_path:
        raise RuntimeError("Go2 highlight marker material readback failed")
    return {
        "enabled": True,
        "robot_prim_path": robot_path,
        "marker_path": marker_path,
        "mode": "base_child_emissive_marker",
        "material_path": material_path,
        "linear_rgb": [float(value) for value in color],
        "emission_scale": float(emission),
        "radius_m": 0.16,
        "base_local_translation_xyz_m": [0.0, 0.0, 0.36],
        "source_robot_materials_modified": False,
        "source_robot_instances_expanded": False,
        "scope": "render marker only; collision, mass, articulation and policy unchanged",
    }


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def tracking_debug(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("tracking_debug") or row.get("base_route_command_before_dynamic_avoidance", {}).get(
        "tracking_debug", {}
    )


def load_replay(path: Path, start_arc: float) -> tuple[list[list[float]], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start_index = next(
        index for index, row in enumerate(rows) if float(tracking_debug(row).get("arc_progress_m", -1.0)) >= start_arc
    )
    selected = rows[start_index:]
    return [list(map(float, row["control_command_body"])) for row in selected], {
        "source": str(path.resolve()),
        "source_sha256": capture.sha256(path.resolve()),
        "source_start_index": start_index,
        "source_start_timestamp_s": float(selected[0]["timestamp_s"]),
        "source_end_timestamp_s": float(selected[-1]["timestamp_s"]),
        "source_command_count": len(selected),
        "source_start_arc_progress_m": float(tracking_debug(selected[0])["arc_progress_m"]),
    }


def closest_route_metrics(point: np.ndarray, route_points: np.ndarray, arc: np.ndarray) -> tuple[float, float]:
    starts = route_points[:-1]
    segments = route_points[1:] - starts
    length_squared = np.sum(segments * segments, axis=1)
    projection_fraction = np.clip(
        np.sum((point[None, :] - starts) * segments, axis=1)
        / np.maximum(length_squared, 1.0e-12),
        0.0,
        1.0,
    )
    projections = starts + projection_fraction[:, None] * segments
    distance = np.linalg.norm(projections - point[None, :], axis=1)
    index = int(np.argmin(distance))
    segment_length = math.sqrt(float(length_squared[index]))
    return (
        float(arc[index] + projection_fraction[index] * segment_length),
        float(distance[index]),
    )


def ghost_overlap_labels(
    robot_xy: np.ndarray,
    ghost_experiment: Any,
    people_xyz: np.ndarray,
    micromobility_states: dict[str, Any] | None,
    traffic_state: dict[str, Any] | None,
) -> list[str]:
    """Conservative geometric labels for intervals excluded from ordinary training."""

    labels: list[str] = []
    point = np.asarray(robot_xy, dtype=np.float64)
    if ghost_experiment is not None:
        for obstacle in ghost_experiment.obstacles:
            delta = point - np.asarray(obstacle.center_xyz[:2], dtype=np.float64)
            yaw = math.radians(float(obstacle.yaw_deg))
            local = np.asarray(
                [
                    math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1],
                    -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1],
                ]
            )
            half = np.asarray(obstacle.dimensions_xyz[:2], dtype=np.float64) * 0.5 + 0.45
            if np.all(np.abs(local) <= half):
                labels.append(f"static:{obstacle.obstacle_id}")
    if len(people_xyz):
        distances = np.linalg.norm(np.asarray(people_xyz)[:, :2] - point, axis=1)
        labels.extend(f"person:{index:02d}" for index in np.flatnonzero(distances <= 0.75))
    if micromobility_states:
        for agent_id, state in micromobility_states.items():
            if np.linalg.norm(state.position_xy - point) <= 1.25:
                labels.append(f"micromobility:{agent_id}")
    if traffic_state is not None:
        for agent in traffic_state.get("agents", []):
            if not agent.get("active_on_road", False):
                continue
            center = np.asarray(agent.get("center_xyz", (math.inf, math.inf, 0.0)))[:2]
            if np.linalg.norm(center - point) <= 2.8:
                labels.append(f"vehicle:{agent.get('agent_id', agent.get('id', 'unknown'))}")
    return labels


def convert_ghost_obstacle_visuals(
    app: Any,
    experiment: Any,
    run_dir: Path,
) -> dict[str, Path]:
    """Convert glTF obstacle visuals to run-local USDs before scene construction.

    Raw ``.glb`` payloads do not compose meshes in this Kit ("Cannot determine
    file format ... SDF_FORMAT_ARGS:target=usd"), so ghost obstacles reuse the
    proven micromobility GLB -> USD conversion and reference the converted USDs.
    """
    obstacles = [
        obstacle
        for obstacle in experiment.obstacles
        if obstacle.visual_asset_relative
        and obstacle.visual_asset_kind == "gltf_payload"
    ]
    if not obstacles:
        return {}
    import omni.kit.asset_converter
    from omni.kit.async_engine import run_coroutine

    converted: dict[str, Path] = {}
    output_root = run_dir / "converted_ghost_visuals"
    for obstacle in obstacles:
        source = (PROJECT_ROOT / obstacle.visual_asset_relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_root / obstacle.obstacle_id / "visual.usd"
        target.parent.mkdir(parents=True, exist_ok=True)
        context = omni.kit.asset_converter.AssetConverterContext()
        context.keep_all_materials = True
        context.create_world_as_default_root_prim = True
        context.use_meter_as_world_unit = True
        task = omni.kit.asset_converter.get_instance().create_converter_task(
            str(source), str(target), None, context
        )
        future = run_coroutine(task.wait_until_finished())
        while not future.done():
            app.update()
        if not future.result() or not target.is_file():
            raise RuntimeError(
                f"ghost visual conversion failed for {obstacle.obstacle_id}: "
                f"{task.get_error_message()}"
            )
        converted[obstacle.obstacle_id] = target
    return converted


def make_trajectory_plot(path: Path, mode: str, route_points: np.ndarray, actual: np.ndarray, status: str, segments=None) -> None:
    width, height, margin = 1100, 700, 70
    image = Image.new("RGB", (width, height), (239, 244, 247))
    draw = ImageDraw.Draw(image)
    draw.text((20, 15), f"Go2 实际分段轨迹 · {mode} · {status}", fill=(20, 29, 36), font=load_font(26, bold=True))
    combined = np.vstack((route_points, actual[:, :2])) if len(actual) else route_points
    low = np.min(combined, axis=0)
    high = np.max(combined, axis=0)
    span = np.maximum(high - low, 1.0)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin - 35) / span[1])

    def project(points: np.ndarray) -> list[tuple[float, float]]:
        return [
            (margin + (value[0] - low[0]) * scale, height - margin - (value[1] - low[1]) * scale)
            for value in points
        ]

    route_xy = project(route_points)
    actual_xy = project(actual[:, :2]) if len(actual) else []
    if len(route_xy) > 1:
        draw.line(route_xy, fill=(46, 196, 126), width=5)
    if len(actual_xy) > 1:
        for i in range(1, len(actual_xy)):
            if segments is None or segments[i] == segments[i-1]:
                draw.line(actual_xy[i-1:i+1], fill=(242, 151, 55), width=5)
        draw.ellipse((actual_xy[0][0] - 7, actual_xy[0][1] - 7, actual_xy[0][0] + 7, actual_xy[0][1] + 7), fill=(50, 125, 210))
        draw.ellipse((actual_xy[-1][0] - 7, actual_xy[-1][1] - 7, actual_xy[-1][0] + 7, actual_xy[-1][1] + 7), fill=(215, 62, 62))
    draw.text((20, height - 35), "绿色=参考路线  橙色=实际轨迹  蓝色=起点  红色=终点", fill=(52, 65, 74), font=load_font(17))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> int:
    # Long USD/PhysX initialization calls can hold the GIL.  Keep an explicit
    # SIGUSR1 stack-dump hook so unattended gates leave actionable evidence
    # instead of appearing as a silent high-CPU hang.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    args = parse_args()
    selected_experience = None
    if args.experience is not None:
        selected_experience = args.experience.resolve()
    elif args.roaming_ghost_config is not None:
        selected_experience = (
            PROJECT_ROOT / "configs" / "isaac45_urbanverse_go2_people_headless.kit"
        ).resolve()
    elif args.mixed_roaming_config is not None:
        # GLB file-format registration is needed before any wrapper opens,
        # including camera-off smoke tests. Default Lab headless lacks it.
        kit_name = 'isaac45_urbanverse_go2_headless.kit' if args.overview_video else 'isaac45_urbanverse_go2_physics_only.kit'
        selected_experience=(PROJECT_ROOT/'configs'/kit_name).resolve()
    if selected_experience is not None and not selected_experience.is_file():
        raise FileNotFoundError(f"Kit experience does not exist: {selected_experience}")
    joint_people_video = (
        args.joint_three_panel_video
        or args.joint_four_panel_video
        or args.roaming_ghost_two_panel_video
    )
    project_native_two_panel = bool(
        args.roaming_ghost_two_panel_video
        and args.mixed_roaming_config is not None
    )
    started = time.perf_counter()
    git_commit_at_launch = run_text(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"])
    git_status_at_launch = run_text(["git", "-C", str(PROJECT_ROOT), "status", "--short"])
    run_dir = args.run_dir.resolve()
    metadata_dir = run_dir / "metadata"
    captures_dir = run_dir / "captures"
    visualizations_dir = run_dir / "visualizations"
    wrapper_dir = run_dir / "wrapper"
    for directory in (metadata_dir, captures_dir, visualizations_dir, wrapper_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary_path = metadata_dir / "summary.json"
    write_json(
        summary_path,
        {
            "status": "running",
            "mode": args.mode,
            "kit_experience": (
                str(selected_experience) if selected_experience is not None else None
            ),
        },
    )
    if selected_experience is not None:
        print(
            "KIT_EXPERIENCE "
            + json.dumps(
                {
                    "path": str(selected_experience),
                    "sha256": capture.sha256(selected_experience),
                    "override": args.experience is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    route_payload = json.loads(args.reference_route.read_text(encoding="utf-8"))
    roaming_ghost_payload = None
    roaming_ghost_config_path = None
    mixed_roaming_payload = None
    mixed_roaming_config_path = None
    roaming_payload = None
    roaming_config_path = None
    roaming_config_sha256_at_load = None
    ghost_mode_enabled = False
    road_sweep_payload = None
    road_sweep_eye_points = None
    project_native_road_sweep = False
    dynamic_agents_ignore_go2 = False
    closed_loop_document=None
    dynamic_contact_metadata=None
    dynamic_contact_runtime=None
    preconverted_project_micro=None
    if args.roaming_ghost_config is not None or args.mixed_roaming_config is not None:
        roaming_config_path = (
            args.mixed_roaming_config or args.roaming_ghost_config
        ).resolve()
        roaming_config_bytes = roaming_config_path.read_bytes()
        roaming_config_sha256_at_load = hashlib.sha256(roaming_config_bytes).hexdigest()
        roaming_payload = json.loads(roaming_config_bytes)
        write_json(metadata_dir/'effective_roaming_config.json',dict(
            source_path=str(roaming_config_path),source_sha256=roaming_config_sha256_at_load,
            git_commit_at_launch=git_commit_at_launch,loaded_config=roaming_payload))
        if args.mixed_roaming_config is not None:
            mixed_roaming_config_path = roaming_config_path
            mixed_roaming_payload = roaming_payload
            dynamic_agents_ignore_go2 = bool(
                roaming_payload.get("go2", {}).get(
                    "dynamic_agents_ignore_go2", False
                )
            )
        else:
            roaming_ghost_config_path = roaming_config_path
            roaming_ghost_payload = roaming_payload

        def resolve_combined(relative: str) -> Path:
            candidate = Path(relative)
            return candidate.resolve() if candidate.is_absolute() else (
                roaming_config_path.parent / candidate
            ).resolve()

        if mixed_roaming_payload is not None and mixed_roaming_payload.get('closed_loops'):
            closed_loop_document=json.loads(resolve_combined(mixed_roaming_payload['closed_loops']).read_text())
            no_lane_runtime=bool(args.traffic_scene_config is not None and
                TrafficSceneConfig.load(args.traffic_scene_config).traffic.get('disabled_reason')=='not_applicable_no_lane')
            if no_lane_runtime and (args.multivehicle_traffic or args.scene10_multivehicle_traffic or
                                    int(mixed_roaming_payload['vehicles']['count'])!=0):
                raise ValueError('No-Lane scene must not enable vehicle runtime')
            if any(not group.get('routes') and not (name=='vehicles' and no_lane_runtime and
                   group.get('status')=='not_applicable_no_lane')
                   for name,group in closed_loop_document['groups'].items()):
                raise ValueError('Closed-loop joint run requires routes for every applicable dynamic group')

        if int(roaming_payload["fixed_seed"]) != int(args.seed):
            raise ValueError(
                "runner seed must equal roaming fixed_seed: "
                f"{args.seed} != {roaming_payload['fixed_seed']}"
            )
        # Resident People and micromobility are a reusable joint-environment
        # capability.  Ghost collision filtering is an optional Go2 experiment,
        # not a prerequisite for enabling those agents.
        ghost_mode_enabled = bool(roaming_payload.get("ghost_experiment"))
        configured_road_sweep = roaming_payload.get("video", {}).get(
            "road_sweep_camera"
        )
        if (
            configured_road_sweep is not None
            and bool(configured_road_sweep.get("enabled", False))
            and args.overview_video
            and not args.disable_road_sweep_video
        ):
            road_sweep_payload = configured_road_sweep
            road_sweep_eye_points = np.asarray(
                road_sweep_payload["eye_points_world_xyz"], dtype=np.float64
            )
            # Validate geometry and view parameters before launching Kit.
            polyline_camera_view(
                0.0,
                road_sweep_eye_points,
                float(road_sweep_payload["speed_mps"]),
                float(road_sweep_payload["pitch_down_deg"]),
                float(road_sweep_payload["lookahead_m"]),
            )
            resolution = road_sweep_payload["output_resolution"]
            if (
                len(resolution) != 2
                or int(resolution[0]) <= 0
                or int(resolution[1]) <= 0
            ):
                raise ValueError(
                    "road_sweep_camera.output_resolution must contain positive width/height"
                )
            if float(road_sweep_payload["speed_mps"]) <= 0.0:
                raise ValueError("road_sweep_camera.speed_mps must be positive")
            if not 0.0 <= float(road_sweep_payload["pitch_down_deg"]) < 90.0:
                raise ValueError(
                    "road_sweep_camera.pitch_down_deg must be in [0, 90)"
                )
            if float(road_sweep_payload["lookahead_m"]) <= 0.0:
                raise ValueError("road_sweep_camera.lookahead_m must be positive")
            project_native_road_sweep = bool(
                mixed_roaming_payload is not None
                and not args.roaming_ghost_two_panel_video
            )
            if project_native_road_sweep:
                args.overview_width = int(resolution[0])
                args.overview_height = int(resolution[1])
                args.overview_focal_length = float(
                    road_sweep_payload["focal_length_mm"]
                )
    maneuver_payload = None
    if args.mode == "three_point":
        if args.maneuver_plan is None:
            raise ValueError("--maneuver-plan is required for three_point mode")
        maneuver_payload = json.loads(args.maneuver_plan.read_text(encoding="utf-8"))
        route_points = np.asarray(maneuver_payload["preview_points_xy"], dtype=np.float64)
        spawn_points = np.asarray(maneuver_payload["approach_points_xy"], dtype=np.float64)
    else:
        route_points = np.asarray(route_payload["points_xy"], dtype=np.float64)
        spawn_points = route_points
    route_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(route_points, axis=0), axis=1))))
    route_yaw = math.atan2(
        float(spawn_points[1, 1] - spawn_points[0, 1]), float(spawn_points[1, 0] - spawn_points[0, 0])
    )
    replay_commands: list[list[float]] | None = None
    replay_meta = None
    if args.mode == "replay":
        if args.replay_trajectory is None:
            raise ValueError("--replay-trajectory is required for replay mode")
        replay_commands, replay_meta = load_replay(args.replay_trajectory.resolve(), args.replay_start_arc)
    duration_s = args.duration_s
    if duration_s is None:
        duration_s = 30.0 if args.mode == "constant" else 300.0 if args.mode in ("route", "three_point") else len(replay_commands) * 0.02
    wrapper_path = wrapper_dir / "scene10_policy_isolation_wrapper.usda"
    if roaming_ghost_payload is not None:
        # Official Recast in Isaac Sim 4.5 only proved reliable when the final
        # opened layer directly references the source scene.  Keep the city as
        # a sibling of the runtime /World so its authored root transform cannot
        # move Isaac Lab's robot/support prims.  Do not add a second host layer:
        # that composition suppresses the auto-rebake event.
        wrapper_path.write_text(
            "#usda 1.0\n"
            "(\n"
            "    defaultPrim = \"World\"\n"
            "    metersPerUnit = 1\n"
            "    upAxis = \"Z\"\n"
            ")\n\n"
            "def Xform \"World\"\n"
            "{\n"
            "}\n\n"
            "def Xform \"UrbanVerseScene\" (\n"
            f"    prepend references = @{args.source_usd.resolve()}@</World>\n"
            ")\n"
            "{\n"
            "}\n",
            encoding="utf-8",
        )
    else:
        wrapper_path.write_text(
            args.wrapper_template.read_text(encoding="utf-8").replace(
                "SOURCE_USD", str(args.source_usd.resolve())
            ),
            encoding="utf-8",
        )
    light_scale_overrides = None
    go2_support_metadata = None
    simulation_app = None
    env = None
    trajectory_handle = None
    three_camera_writer = None
    three_camera_definitions = None
    overlap_recorder = None
    overlap_labeler = None
    video_writer = None
    road_sweep_video_writer = None
    go2_highlight_metadata = None
    joint_render_bridge_settings = None
    result_code = 1
    try:
        launcher = None
        if roaming_ghost_payload is not None:
            # Recast 106.4's stage listener is disrupted by AppLauncher's
            # post-start PhysX/settings patch phase.  Start the proven combined
            # experience directly, bake NavMesh, then import Isaac Lab below.
            # This remains one Kit process and one final USD stage.
            import builtins
            from isaacsim import SimulationApp

            builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False
            simulation_app = SimulationApp(
                {
                    "headless": True,
                    "renderer": "RayTracedLighting",
                    "width": args.overview_width,
                    "height": args.overview_height,
                    "active_gpu": args.gpu,
                    "physics_gpu": args.gpu,
                    "multi_gpu": False,
                    "max_gpu_count": 1,
                    "create_new_stage": False,
                },
                experience=str(selected_experience),
            )
        else:
            from isaaclab.app import AppLauncher

            launcher_kwargs = {
                "headless": True,
                "enable_cameras": args.overview_video,
                "device": f"cuda:{args.gpu}",
                "width": args.overview_width,
                "height": args.overview_height,
                "multi_gpu": False,
            }
            if args.overview_video:
                launcher_kwargs["renderer"] = "RayTracedLighting"
                launcher_kwargs["experience"] = str(
                    PROJECT_ROOT / "configs" / "isaac45_urbanverse_go2_headless.kit"
                )
            if selected_experience is not None:
                launcher_kwargs["experience"] = str(selected_experience)
            launcher = AppLauncher(**launcher_kwargs)
            simulation_app = launcher.app

        import carb
        import omni.usd

        if roaming_ghost_payload is not None:
            # The official-People composition starts SimulationApp directly so
            # Recast can finish before Isaac Lab imports.  That intentionally
            # bypasses AppLauncher, but AppLauncher is also what normally turns
            # on the headless off-screen path and PhysX-Fabric transform bridge
            # when ``enable_cameras=True``.  Without these settings, post-reset
            # Replicator products see USD-authored People/traffic while the RTX
            # scene keeps the articulated Go2 at its initial Fabric pose.
            #
            # Configure the same bridge before SimulationContext is created,
            # without creating a camera or render product before env.reset().
            # Keeping camera creation deferred avoids the proven reset lock.
            joint_settings = carb.settings.get_settings()
            joint_settings.set_bool(
                "/isaaclab/cameras_enabled", bool(args.overview_video)
            )
            joint_settings.set_bool(
                "/isaaclab/render/offscreen", bool(args.overview_video)
            )
            joint_settings.set_bool("/isaaclab/render/active_viewport", False)
            joint_settings.set_bool(
                "/physics/fabricUpdateTransformations", bool(args.overview_video)
            )
            joint_render_bridge_settings = {
                "configuration_stage": "after SimulationApp startup, before Recast and Isaac Lab SimulationContext",
                "camera_creation_stage": "after env.reset",
                "requested_overview_video": bool(args.overview_video),
                "isaaclab_cameras_enabled": bool(
                    joint_settings.get("/isaaclab/cameras_enabled")
                ),
                "isaaclab_offscreen_render": bool(
                    joint_settings.get("/isaaclab/render/offscreen")
                ),
                "isaaclab_active_viewport": bool(
                    joint_settings.get("/isaaclab/render/active_viewport")
                ),
                "physics_update_to_usd": bool(
                    joint_settings.get("/physics/updateToUsd")
                ),
                "physics_fabric_update_transformations": bool(
                    joint_settings.get("/physics/fabricUpdateTransformations")
                ),
            }
            print(
                "JOINT_RENDER_BRIDGE_CONFIG "
                + json.dumps(joint_render_bridge_settings, sort_keys=True),
                flush=True,
            )

        official_extensions = (
            enable_official_people_runtime(simulation_app)
            if roaming_ghost_payload is not None
            else None
        )
        for _ in range(3):
            simulation_app.update()

        light_path_scales: dict[str, float] = {}
        for item in args.collection_light_path_scale:
            if "=" not in item:
                raise ValueError("--collection-light-path-scale must use PRIM_PATH=SCALE")
            prim_path, raw_scale = item.rsplit("=", 1)
            if (
                roaming_ghost_payload is not None
                and prim_path == "/UrbanVerseAsset/DomeLight_04"
            ):
                # This is the legacy Scene10 runner default.  Scene-portable
                # official-Recast compositions retain the source's native
                # /World and must not require a Scene10-only light prim.
                continue
            if prim_path in light_path_scales:
                raise ValueError(f"duplicate collection light path scale: {prim_path}")
            light_path_scales[prim_path] = float(raw_scale)
        light_scale_overrides = capture.author_all_light_scale(
            wrapper_path,
            args.collection_light_scale,
            dome_scale=args.collection_dome_light_scale,
            distant_scale=args.collection_distant_light_scale,
            sphere_scale=args.collection_sphere_light_scale,
            path_scales=light_path_scales,
            source_dome_background_visible=args.source_dome_background_visible,
            source_dome_texture_enabled=args.source_dome_texture_enabled,
        )
        write_json(metadata_dir / "collection_light_scale.json", light_scale_overrides)
        print(
            "COLLECTION_LIGHT_SCALE "
            + json.dumps(
                {
                    "requested_scales": light_scale_overrides["requested_scales"],
                    "path_scales": light_scale_overrides["path_scales"],
                    "adjusted_count": light_scale_overrides["adjusted_count"],
                    "type_counts": light_scale_overrides["type_counts"],
                    "source_dome_background_visible": light_scale_overrides[
                        "source_dome_background_visible"
                    ],
                    "source_dome_texture_enabled": light_scale_overrides[
                        "source_dome_texture_enabled"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        preauthored_traffic_visuals = []
        portable_scene = TrafficSceneConfig.load(args.traffic_scene_config) if args.traffic_scene_config else None
        pedestrian_config = None
        preauthored_people = []
        project_approved_walkable_union = None
        project_walkable_regions = None
        project_micromobility_regions = None
        project_roaming_assignments = ()
        if (
            (args.multivehicle_traffic or args.scene10_multivehicle_traffic)
            and args.traffic_registry is not None
        ):
            preauthored_traffic_visuals = author_portable_traffic_visuals(
                wrapper_path,
                args.traffic_registry,
                args.traffic_vehicle_count,
                **({'asset_ids':portable_scene.traffic['asset_ids']} if portable_scene and portable_scene.traffic.get('asset_ids') else {}),
                author_root=(
                    "/World"
                    if roaming_ghost_payload is not None
                    else "/UrbanVerseAsset"
                ),
                mount_root=(
                    "/World"
                    if roaming_ghost_payload is not None
                    else "/World/ground/terrain"
                ),
            )
        if args.pedestrian_config is not None:
            pedestrian_config, preauthored_people = author_official_people(
                wrapper_path, args.pedestrian_config
            )
        if mixed_roaming_payload is not None:
            if args.pedestrian_config is not None:
                raise ValueError(
                    "--pedestrian-config and --mixed-roaming-config are exclusive"
                )
            project_approved_walkable_union = WalkableRegions.load(
                resolve_combined(mixed_roaming_payload["walkable_regions"])
            )
            interaction_bands = mixed_roaming_payload.get("interaction_bands", {})
            people_semantics = tuple(
                value.lower()
                for value in interaction_bands.get(
                    "people_semantics",
                    project_approved_walkable_union.config.allowed_semantics,
                )
            )
            micromobility_semantics = tuple(
                value.lower()
                for value in interaction_bands.get(
                    "micromobility_semantics",
                    project_approved_walkable_union.config.allowed_semantics,
                )
            )
            if set(people_semantics) != set(micromobility_semantics):
                raise ValueError(
                    "project mixed roaming requires People and micromobility to use "
                    "the same approved semantic union"
                )
            project_walkable_regions = project_approved_walkable_union.semantic_subset(
                people_semantics
            )
            project_micromobility_regions = project_walkable_regions
            if closed_loop_document is not None:
                from dataclasses import replace
                from urbanverse.dynamic_agents.navigation.loop_population import distribute,people_assignments,reduced_allocation
                project_micromobility_regions=WalkableRegions(replace(project_walkable_regions.config,clearance_m=0.))
            people_payload = mixed_roaming_payload["people"]
            people_assets_root_value = Path(people_payload["assets_root"])
            project_people_assets_root = (
                people_assets_root_value.resolve()
                if people_assets_root_value.is_absolute()
                else (PROJECT_ROOT / people_assets_root_value).resolve()
            )
            people_assets = tuple(
                project_people_assets_root / relative
                for relative in people_payload["assets"]
            )
            project_people_loops={}
            if closed_loop_document is not None:
                project_people_loops=distribute(closed_loop_document['groups']['people']['routes'],
                    int(people_payload['count']),'Pedestrian',project_walkable_regions,
                    bool(mixed_roaming_payload.get('reverse_alternate_loops',False)),
                    reduced_allocation(closed_loop_document.get('population_proposal',{}).get('people',{}).get('per_route'),int(people_payload['count'])),
                    float(people_payload.get('minimum_spawn_separation_m',1.2)))
            project_roaming_assignments = people_assignments(project_people_loops,people_assets,project_walkable_regions.config.ground_z_m) if project_people_loops else plan_roaming_assignments(
                project_walkable_regions,
                people_assets,
                count=int(people_payload["count"]),
                seed=args.seed,
                minimum_spawn_separation_m=float(
                    people_payload.get("minimum_spawn_separation_m", 1.2)
                ),
                minimum_initial_trip_m=float(people_payload["minimum_trip_m"]),
                maximum_initial_trip_m=float(people_payload["maximum_trip_m"]),
            )
            people_settings = dict(people_payload)
            people_settings["seed"] = args.seed
            if project_people_loops:
                people_settings['closed_paths_xy']={name:item['xy'].tolist() for name,item in project_people_loops.items()}
            pedestrian_config = build_project_people_payload(
                project_walkable_regions,
                project_roaming_assignments,
                project_people_assets_root,
                people_settings,
            )
            pedestrian_config, preauthored_people = author_people_payload(
                wrapper_path, pedestrian_config
            )
            write_json(
                metadata_dir / "project_people_authored_config.json",
                pedestrian_config,
            )

        # Recast must complete before parse_env_cfg imports and initializes the
        # Isaac Lab simulation stack.  Open the final host stage now, while the
        # process still matches the proven standalone official-People order.
        prebaked_official_people_runtime = None
        prebaked_approved_walkable_union = None
        prebaked_walkable_regions = None
        prebaked_micromobility_regions = None
        prebaked_navmesh_metadata = None
        prebaked_people_assets_root = None
        prebaked_roaming_assignments = ()
        prebaked_roaming_waypoint_loops = None
        prebaked_official_setup_metadata = None
        prebaked_official_play_updates = None
        if roaming_ghost_payload is not None:
            from omni.anim.navigation.core import NavMeshSettings
            import omni.timeline

            preload_stage_path = wrapper_path
            context = omni.usd.get_context()
            context.open_stage(
                str(preload_stage_path),
                None,
                omni.usd.UsdContextInitialLoadSet.LOAD_ALL,
            )
            for _ in range(8):
                simulation_app.update()
            preload_stage = context.get_stage()
            if preload_stage is None:
                raise RuntimeError("failed to open Scene09 roaming preload stage")
            preload_stage.SetEditTarget(preload_stage.GetSessionLayer())
            preload_timeline = omni.timeline.get_timeline_interface()
            preload_timeline.stop()
            for _ in range(3):
                simulation_app.update()
            if preload_timeline.is_playing():
                raise RuntimeError("pre-physics NavMesh stage did not stop")
            prebaked_approved_walkable_union = WalkableRegions.load(
                resolve_combined(roaming_ghost_payload["walkable_regions"])
            )
            interaction_bands = roaming_ghost_payload.get("interaction_bands", {})
            people_semantics = interaction_bands.get("people_semantics")
            micromobility_semantics = interaction_bands.get(
                "micromobility_semantics"
            )
            prebaked_walkable_regions = (
                prebaked_approved_walkable_union.semantic_subset(people_semantics)
                if people_semantics
                else prebaked_approved_walkable_union
            )
            prebaked_micromobility_regions = (
                prebaked_approved_walkable_union.semantic_subset(
                    micromobility_semantics
                )
                if micromobility_semantics
                else prebaked_approved_walkable_union
            )
            nav_settings = carb.settings.get_settings()
            nav_settings.set(NavMeshSettings.CACHE_ENABLED_SETTING_PATH, False)
            nav_settings.set(NavMeshSettings.AUTO_REBAKE_SETTING_PATH, True)
            prebaked_navmesh_metadata = author_approved_navmesh_surface(
                preload_stage,
                prebaked_walkable_regions,
                exclusion_strategy="temporary_visibility",
                source_scene_path="/UrbanVerseScene",
                mesh_path="/ApprovedWalkingSurface",
                volume_path="/ApprovedWalkingNavMeshVolumes",
            )
            people_payload = roaming_ghost_payload["official_people"]
            people_assets_root_value = Path(people_payload["assets_root"])
            prebaked_people_assets_root = (
                people_assets_root_value.resolve()
                if people_assets_root_value.is_absolute()
                else (PROJECT_ROOT / people_assets_root_value).resolve()
            )
            prebaked_official_people_runtime = OfficialPeopleRuntime(
                simulation_app,
                preload_stage,
                prebaked_people_assets_root,
                characters_parent="/Characters",
            )
            prebaked_official_people_runtime.configure_avoidance_radius(
                float(people_payload.get("avoidance_radius_m", 0.65))
            )
            prebaked_navmesh_metadata.update(
                prebaked_official_people_runtime.wait_for_navmesh(
                    navmesh_timeout_updates=1800
                )
            )
            nav_settings.set(NavMeshSettings.AUTO_REBAKE_SETTING_PATH, False)
            people_assets = tuple(
                prebaked_people_assets_root / relative
                for relative in people_payload["assets"]
            )
            prebaked_roaming_assignments = plan_roaming_assignments(
                prebaked_walkable_regions,
                people_assets,
                count=int(people_payload["count"]),
                seed=args.seed,
                minimum_initial_trip_m=float(
                    people_payload.get("minimum_trip_m", 6.0)
                ),
                maximum_initial_trip_m=float(
                    people_payload.get("maximum_target_leg_m", 18.0)
                ),
                multi_resident_minimum_width_m=(
                    None
                    if people_payload.get("multi_resident_minimum_width_m") is None
                    else float(people_payload["multi_resident_minimum_width_m"])
                ),
            )
            prebaked_roaming_logical_waypoint_loops = build_roaming_waypoint_loops(
                prebaked_walkable_regions,
                prebaked_roaming_assignments,
                np.random.default_rng(args.seed + 17017),
                ring_points=int(people_payload.get("target_cycle_points", 9)),
                radius_min_m=float(
                    people_payload.get("target_cycle_radius_min_m", 8.0)
                ),
                radius_max_m=float(
                    people_payload.get("target_cycle_radius_max_m", 18.0)
                ),
                maximum_leg_m=float(
                    people_payload.get("maximum_target_leg_m", 30.0)
                ),
                minimum_leg_m=float(
                    people_payload.get("minimum_target_leg_m", 5.0)
                ),
                partition_shared_components=bool(
                    people_payload.get("partition_shared_components", True)
                ),
            )
            missing_cycles = sorted(
                item.name
                for item in prebaked_roaming_assignments
                if item.name not in prebaked_roaming_logical_waypoint_loops
            )
            if missing_cycles:
                raise RuntimeError(
                    "could not build bounded roaming target cycles for residents: "
                    + ", ".join(missing_cycles)
                )
            navigation_subleg_max_m = people_payload.get(
                "maximum_navigation_subleg_m"
            )
            prebaked_roaming_waypoint_loops = (
                expand_waypoint_loops_with_grid_paths(
                    prebaked_walkable_regions,
                    prebaked_roaming_assignments,
                    prebaked_roaming_logical_waypoint_loops,
                    maximum_navigation_leg_m=float(navigation_subleg_max_m),
                )
                if navigation_subleg_max_m is not None
                else prebaked_roaming_logical_waypoint_loops
            )
            prebaked_roaming_assignments = align_initial_targets_to_waypoint_loops(
                prebaked_roaming_assignments,
                prebaked_roaming_waypoint_loops,
            )
            # Character staging and Animation Graph setup can rebuild PhysX
            # internals.  Complete that work before Isaac Lab creates tensor
            # views; doing it after gym.make() makes the next env.step access
            # stale articulation handles and segfault in PhysX 106.5.
            official_character_metadata = (
                prebaked_official_people_runtime.load_characters(
                    assignments_to_specs(prebaked_roaming_assignments),
                    metadata_dir / "official_people_initial_commands.txt",
                )
            )
            official_pre_restore_routes = (
                prebaked_official_people_runtime.validate_routes()
            )
            if not all(row["resolved"] for row in official_pre_restore_routes):
                raise RuntimeError(
                    "one or more resident routes do not resolve after NavMesh bake"
                )
            # ``play`` performs Kit updates while Animation Graph creates its
            # native CharacterManager handles.  Those updates must also finish
            # before Isaac Lab creates PhysX tensor views.  Doing only
            # ``setup_all_characters`` here and calling ``play`` after
            # ``env.reset`` leaves stale articulation handles and the first
            # env.step can segfault in set_dof_actuation_forces.
            prebaked_official_play_updates = prebaked_official_people_runtime.play(
                time_codes_per_second=50.0,
                target_framerate_hz=50.0,
                timeout_updates=180,
                restart_timeline=False,
            )
            # Hand ownership of the shared timeline to Isaac Lab while it
            # constructs SimulationContext.  ``pause`` preserves the staged
            # Animation Graph and CharacterManager handles; unlike ``stop`` it
            # does not dispatch the destructive render-on-stop/reset path.
            # Keeping the timeline playing here makes gym.make() wait forever
            # for exclusive physics-scene initialization.
            if prebaked_official_people_runtime.timeline is not None:
                prebaked_official_people_runtime.timeline.pause()
                prebaked_official_people_runtime.timeline.commit()
            # Keep the full city hidden until native CharacterManager handles
            # exist. Restoring it before play() couples the first Animation
            # Graph update to whole-city RTX/MDL material compilation and can
            # leave that update CPU-bound for many minutes after a cold cache.
            # Visibility restoration itself changes neither Recast polygons
            # nor CharacterManager handles; Isaac Lab performs the subsequent
            # renderer warm-up after it owns the paused timeline.
            prebaked_navmesh_metadata["post_bake"] = finalize_approved_navmesh_surface(
                preload_stage,
                mesh_path="/ApprovedWalkingSurface",
                volume_path="/ApprovedWalkingNavMeshVolumes",
                scene_visibility_records=prebaked_navmesh_metadata.get(
                    "hidden_scene_branches"
                ),
            )
            # Recast/Kit may leave the approved helper or a NavMesh volume
            # selected after baking.  RTX captures then contain bright green
            # editor-selection outlines even though the helper itself is
            # invisible.  Clear only UI selection state; this does not modify
            # USD geometry, navigation data, or collision schemas.
            selection = context.get_selection()
            selected_before_clear = list(selection.get_selected_prim_paths())
            selection.set_selected_prim_paths([], False)
            prebaked_navmesh_metadata["selection_cleanup"] = {
                "selected_before_clear": selected_before_clear,
                "selected_after_clear": list(selection.get_selected_prim_paths()),
            }
            official_route_validation = (
                prebaked_official_people_runtime.validate_routes()
            )
            if not all(row["resolved"] for row in official_route_validation):
                raise RuntimeError(
                    "one or more resident routes fail after scene restore"
                )
            prebaked_official_setup_metadata = {
                **official_character_metadata,
                **prebaked_navmesh_metadata,
                "pre_restore_route_validation": official_pre_restore_routes,
                "post_bake_route_validation": official_route_validation,
                "initialization_order": (
                    "open final stage, bake NavMesh, stage/setup official People, "
                    "initialize CharacterManager while the city remains hidden, "
                    "pause the timeline, restore city visibility, then let Isaac "
                    "Lab create PhysX tensor views and warm RTX"
                ),
            }

        # Import the Isaac Lab/PhysX/Replicator stack only after Recast has
        # copied its approved navigation surface.  Importing these modules
        # earlier changes the stage/timeline services enough that Recast 106.4
        # no longer receives its auto-rebake event.
        import gymnasium as gym
        if args.traffic_scene_config is not None:
            from urbanverse.dynamic_agents.navigation.mesh_runtime import author_cleanup
            cleanup=author_cleanup(wrapper_path,TrafficSceneConfig.load(args.traffic_scene_config))
            if cleanup is not None:
                write_json(metadata_dir/'mesh_cleanup.json',cleanup)
        if closed_loop_document is not None and mixed_roaming_payload.get('go2',{}).get('collision_recovery',{}).get('enabled'):
            from urbanverse.dynamic_agents.integration.dynamic_contact import author as author_contact
            scene_spec=TrafficSceneConfig.load(args.traffic_scene_config)
            dynamic_contact_metadata=author_contact(wrapper_path,scene_spec.vehicle_catalog,
                {key:int(mixed_roaming_payload[key]['count']) for key in ['vehicles','people','micromobility']},
                load_micromobility_catalog(resolve_combined(mixed_roaming_payload['micromobility']['catalog'])),
                scene_spec.fallback_ground_z_m,project_walkable_regions.config.ground_z_m)
            write_json(metadata_dir/'dynamic_contact_proxies.json',dynamic_contact_metadata)
        import omni.physx
        import torch
        from isaacsim.core.utils.extensions import enable_extension
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        enable_extension("omni.kit.asset_converter")
        task = navigation.CONTROLLER_PROFILES["flat"]["task"]
        cfg = parse_env_cfg(task, device=f"cuda:{args.gpu}", num_envs=1)
        # Isaac Lab 4.5 points the stock Go2 configuration at a remote HTTP
        # asset.  Kit's omni.client does not consistently inherit the host
        # proxy settings, so a cleared cache can make an otherwise offline-
        # reproducible run fail at gym.make().  Prefer the maintained local
        # Isaac 4.5 asset mirror when it is present; this is scene-independent.
        local_go2_usd = (
            PROJECT_ROOT
            / "data/isaacsim_assets_4_5/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
        )
        configured_go2_usd = str(cfg.scene.robot.spawn.usd_path)
        if local_go2_usd.is_file():
            cfg.scene.robot.spawn.usd_path = str(local_go2_usd.resolve())
        write_json(
            metadata_dir / "go2_asset_resolution.json",
            {
                "configured_usd": configured_go2_usd,
                "resolved_usd": str(cfg.scene.robot.spawn.usd_path),
                "local_mirror_used": bool(local_go2_usd.is_file()),
            },
        )
        deferred_runtime_render_interval: int | None = None
        if preauthored_traffic_visuals:
            configure_preauthored_traffic_visuals(cfg, preauthored_traffic_visuals)
        if preauthored_people:
            configure_preauthored_people(cfg, preauthored_people)
        ghost_experiment = None
        ghost_obstacle_metadata = None
        if ghost_mode_enabled:
            ghost_path = resolve_combined(roaming_ghost_payload["ghost_experiment"])
            ghost_experiment = load_experiment_config(ghost_path)
            # Camera-off A/B gates validate collision behavior, not appearance.
            # Do not start GLB conversion workers in those runs: large converter
            # writes can still be active when InteractiveScene imports the city
            # USD and block stage composition indefinitely.  Camera-enabled
            # runs currently convert the reviewed visuals; the definitive path
            # promotes them to a durable cache before the formal run.
            ghost_visual_overrides = (
                convert_ghost_obstacle_visuals(simulation_app, ghost_experiment, run_dir)
                if args.overview_video
                else {}
            )
            # The A/B normal leg spawns the identical proxies but with no Go2
            # world filter, so they physically block the robot and provide the
            # "normal collides" baseline for the ghost pass-through contrast.
            ghost_obstacle_metadata = configure_experiment_obstacles(
                cfg,
                ghost_experiment,
                "normal" if args.ghost_ab_normal else "passthrough",
                PROJECT_ROOT if args.overview_video else None,
                visual_usd_overrides=ghost_visual_overrides,
            )
        go2_support_metadata = configure_go2_support_corridor(
            cfg,
            route_payload,
            ghost_passthrough=(
                ghost_mode_enabled and not args.ghost_ab_normal
            ),
            isolate_source_scene_collisions=bool(
                roaming_ghost_payload is not None
                and roaming_ghost_payload.get("go2", {}).get(
                    "isolate_source_scene_collisions", False
                )
                and not ghost_mode_enabled
            ),
        )
        if go2_support_metadata is not None:
            write_json(metadata_dir / "go2_runtime_physics_support.json", go2_support_metadata)
            print("GO2_RUNTIME_SUPPORT " + json.dumps(go2_support_metadata, sort_keys=True), flush=True)
        capture.configure_environment(
            cfg,
            wrapper_path,
            {
                "spawn_position": [float(spawn_points[0, 0]), float(spawn_points[0, 1]), float(route_payload["ground_z"]) + 0.4],
                "spawn_yaw_rad": route_yaw,
            },
            None,
            num_envs=1,
        )
        if route_payload.get("live_route_preflight", {}).get("enabled"):
            cfg.sim.enable_scene_query_support = True
        if args.policy_kind in ("robot_lab", "himloco"):
            from urbanverse.dynamic_agents.navigation.external_go2_policy import configure_external_go2_environment
            configure_external_go2_environment(cfg, args.external_policy_source, args.policy_kind)
        # The locomotion task enables its command marker visualization by
        # default.  Those markers are useful in an interactive Isaac Lab
        # session, but their long green rays are real renderable debug geometry
        # and therefore leak into the two-panel evidence video.  The route
        # controller and command values are unchanged when only this visualizer
        # is disabled.
        if getattr(cfg, "commands", None) is not None and getattr(
            cfg.commands, "base_velocity", None
        ) is not None:
            cfg.commands.base_velocity.debug_vis = False
        # Import unconditionally (not only when recording): these modules pull in
        # the Replicator/camera extensions whose omni.graph SDG-pipeline nodes must
        # be registered before gym.make() calls rep.set_global_seed().  Skipping
        # them in camera-off A/B runs made the SDG graph creation fail with
        # "Failed to wrap graph in node ... /Replicator/SDGPipeline".
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg

        if args.go2_three_camera:
            from types import SimpleNamespace
            three_camera_definitions = capture.add_navigation_camera_sensors(cfg, SimpleNamespace(
                capture_fps=args.overview_fps, sensor_width=args.three_camera_width,
                sensor_height=args.three_camera_height, lightweight_three_panel=False,
                strict_document_calibration=True, allow_scaled_document_calibration=True,
                omit_motion_vectors=True), onboard_only=True)

        if args.overview_video:

            def camera_cfg(
                prim_name: str, focal_length: float | None = None
            ) -> CameraCfg:
                return CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
                    update_period=1.0 / args.overview_fps,
                    height=args.overview_height,
                    width=args.overview_width,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=(
                            args.overview_focal_length
                            if focal_length is None
                            else focal_length
                        ),
                        horizontal_aperture=20.955,
                        clipping_range=(0.02, 1000.0),
                        focus_distance=30.0,
                        f_stop=0.0,
                    ),
                    offset=CameraCfg.OffsetCfg(convention="world"),
                    update_latest_camera_pose=True,
                )

            # The roaming ghost deliverable contains only the Go2-follow and
            # resident-person-follow panels.  Do not instantiate the legacy
            # overview render product: even when its pixels are discarded it
            # still incurs a full RTX render on every captured frame.
            if not args.roaming_ghost_two_panel_video:
                cfg.scene.go2_traffic_overview = camera_cfg("TrafficOverviewCamera")
                if args.go2_front_pinhole:
                    cfg.scene.go2_traffic_overview.prim_path = "{ENV_REGEX_NS}/Robot/base/FrontPinholeCamera"
                    cfg.scene.go2_traffic_overview.offset = CameraCfg.OffsetCfg(
                        pos=(0.35, 0.0, 0.10), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
                    )
            if joint_people_video and (
                not args.roaming_ghost_two_panel_video or project_native_two_panel
            ):
                cfg.scene.go2_traffic_go2_follow = camera_cfg("TrafficGo2FollowCamera")
                cfg.scene.go2_traffic_person_follow = camera_cfg(
                    "TrafficPersonFollowCamera"
                )
            if args.joint_four_panel_video:
                cfg.scene.go2_traffic_scene_global = camera_cfg(
                    "TrafficSceneGlobalCamera", args.scene_global_focal_length
                )
            # Official Animation Graph People require a Kit/render update at
            # every 50 Hz control step.  The requested video FPS controls only
            # which RGB frames are encoded; lowering this render interval to
            # the video cadence freezes the official characters even though
            # Go2, traffic, and micromobility continue to advance.
            cfg.sim.render_interval = cfg.decimation
            if mixed_roaming_payload is not None:
                # Project-controlled animation is stepped explicitly; it does
                # not depend on the official Animation Graph's 50 Hz update.
                # Keep physics/policy cadence unchanged and render at video FPS.
                cfg.sim.render_interval = max(cfg.decimation, round(1.0 / (cfg.sim.dt * args.overview_fps)))
        cfg.episode_length_s = float(duration_s) + 20.0
        if args.ghost_ab_headless:
            # Camera-off A/B runs have no Replicator render product, so Isaac
            # Lab's env seed() -- which calls rep.set_global_seed() -- crashes
            # with "Failed to wrap graph in node ... /Replicator/SDGPipeline".
            # Seed the RNGs directly instead; Replicator seeding only matters
            # for rendering/domain-randomization, which camera-off runs skip.
            cfg.seed = None
            import isaacsim.core.utils.torch as torch_utils

            torch_utils.set_seed(args.seed)
        else:
            cfg.seed = args.seed

        if roaming_ghost_payload is not None:
            # The city is already mounted in the live stage.  A second terrain
            # import would duplicate it and invalidate both NavMesh coordinates
            # and obstacle identities.  InteractiveScene supports terrain=None
            # and derives the single environment origin at (0, 0, 0).
            cfg.scene.terrain = None
            # Official People Animation Graph is evaluated from Kit/render
            # updates, not from PhysX tensor stepping alone.  Camera-off gates
            # previously let Go2 and micromobility advance while residents
            # remained frozen.  Ask Isaac Lab to perform one render/update per
            # control step even without a render product; this advances the
            # official graph without adding an uncontrolled physics step.
            if not args.overview_video:
                cfg.sim.render_interval = cfg.decimation

        print(
            "JOINT_ENV_INIT phase=gym_make "
            f"cameras={bool(args.overview_video)} people={len(prebaked_roaming_assignments)}",
            flush=True,
        )
        env = gym.make(task, cfg=cfg)
        if closed_loop_document is not None:
            # gym.make already creates tensor views internally. Pause rather
            # than stop: STOP invokes Isaac Lab's app-shutdown callback and
            # invalidates those views. No uncontrolled physics during conversion.
            import omni.timeline
            timeline = omni.timeline.get_timeline_interface()
            timeline.pause()
            print("JOINT_ENV_INIT phase=convert_micro_before_reset", flush=True)
            preconverted_project_micro=convert_micromobility_assets(simulation_app,
                load_micromobility_catalog(resolve_combined(mixed_roaming_payload['micromobility']['catalog'])),
                PROJECT_ROOT/'data',run_dir/'converted_micromobility')
            if timeline.is_playing():
                raise RuntimeError('Micromobility conversion unexpectedly started physics')
            timeline.play()
            timeline.commit()
            if not timeline.is_playing():
                raise RuntimeError('Failed to resume timeline after micro conversion')
        print("JOINT_ENV_INIT phase=env_reset", flush=True)
        observations, _ = env.reset()
        print("JOINT_ENV_INIT phase=ready", flush=True)
        unwrapped = env.unwrapped
        if closed_loop_document is not None:
            from urbanverse.dynamic_agents.navigation.mesh_runtime import audit_loaded_meshes
            mesh_loaded=audit_loaded_meshes(omni.usd.get_context().get_stage(),TrafficSceneConfig.load(args.traffic_scene_config))
            write_json(metadata_dir/'mesh_runtime_admission.json',mesh_loaded)
            if not mesh_loaded['passed']:raise RuntimeError('Live scene does not match authoritative mesh inventory: '+json.dumps(mesh_loaded))
        if joint_render_bridge_settings is not None:
            joint_settings = carb.settings.get_settings()
            joint_render_bridge_settings["post_reset"] = {
                "simulation_render_mode": unwrapped.sim.render_mode.name,
                "fabric_enabled": bool(unwrapped.sim.is_fabric_enabled()),
                "isaaclab_rtx_sensors": bool(
                    joint_settings.get("/isaaclab/render/rtx_sensors")
                ),
                "physics_update_to_usd": bool(
                    joint_settings.get("/physics/updateToUsd")
                ),
                "physics_fabric_update_transformations": bool(
                    joint_settings.get("/physics/fabricUpdateTransformations")
                ),
            }
            write_json(
                metadata_dir / "fabric_render_bridge.json",
                joint_render_bridge_settings,
            )
            print(
                "JOINT_RENDER_BRIDGE_READY "
                + json.dumps(joint_render_bridge_settings["post_reset"], sort_keys=True),
                flush=True,
            )
        if deferred_runtime_render_interval is not None:
            unwrapped.cfg.sim.render_interval = deferred_runtime_render_interval
            print(
                "JOINT_RENDER_CADENCE "
                f"init_interval={cfg.decimation} "
                f"runtime_interval={deferred_runtime_render_interval}",
                flush=True,
            )
        if os.environ.get("URBANVERSE_CAPTURE_ONLY_RENDER", "0") == "1":
            if mixed_roaming_payload is None or not args.overview_video:
                raise ValueError("Capture-only rendering requires project mixed agents and capture enabled")
            # Diagnostic opt-in: the capture block renders after env.step and
            # camera pose updates. Avoid a second render inside the physics loop.
            # No physics cadence, sensor registration, or static collision change.
            previous_interval = int(unwrapped.cfg.sim.render_interval)
            unwrapped.cfg.sim.render_interval = int(
                (duration_s + 120.0) / unwrapped.physics_dt
            ) + int(unwrapped._sim_step_counter) + 1
            write_json(metadata_dir / "render_schedule.json", {
                "mode": "capture_only_diagnostic",
                "previous_physics_render_interval": previous_interval,
                "physics_render_interval": unwrapped.cfg.sim.render_interval,
                "capture_fps": args.overview_fps,
                "physics_dt": unwrapped.physics_dt,
                "scope": "diagnostic hypothesis, not a proven graph-stall fix",
            })
            print("RTX_DIAGNOSTIC render_schedule=capture_only", flush=True)
        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene["contact_forces"]
        dt = float(unwrapped.step_dt)
        official_people_collision_audit = None
        if roaming_ghost_payload is not None:
            # NVIDIA IRA People are navigation/animation actors.  Their stock
            # assets are expected to have no PhysX CollisionAPI shapes; this is
            # what makes Go2↔People physical pass-through independent of the
            # much larger /Characters skeleton hierarchy.  Audit the composed
            # runtime stage instead of assuming every future People asset has
            # the same authoring convention.
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            character_root = stage.GetPrimAtPath("/Characters")
            character_collider_paths: list[str] = []
            if character_root.IsValid():
                for descendant in Usd.PrimRange(
                    character_root, Usd.TraverseInstanceProxies()
                ):
                    if descendant.HasAPI(UsdPhysics.CollisionAPI):
                        character_collider_paths.append(str(descendant.GetPath()))
            official_people_collision_audit = {
                "root": "/Characters",
                "root_valid": bool(character_root.IsValid()),
                "collision_api_count": len(character_collider_paths),
                "collision_api_prim_paths": character_collider_paths,
                "go2_passthrough_basis": (
                    "official People have no PhysX collision shapes; navigation "
                    "avoidance remains active while Go2 is omitted from its inputs"
                ),
                "passed": bool(
                    character_root.IsValid() and not character_collider_paths
                ),
            }
            write_json(
                metadata_dir / "official_people_collision_audit.json",
                official_people_collision_audit,
            )
            print(
                "OFFICIAL_PEOPLE_COLLISION_AUDIT "
                + json.dumps(official_people_collision_audit, sort_keys=True),
                flush=True,
            )
            if character_collider_paths:
                raise RuntimeError(
                    "official People assets unexpectedly contain PhysX colliders; "
                    "Go2 ghost pass-through is not proven"
                )
        if go2_support_metadata is not None:
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            cache = UsdGeom.XformCache()
            scene_query = omni.physx.get_physx_scene_query_interface()
            readback = []
            for pattern in go2_support_metadata["segments"]:
                path = pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
                prim = stage.GetPrimAtPath(path)
                # A custom collidable mesh is authored directly on ``path``.
                # Author the USD visibility explicitly after composition; this
                # hides only rendering while PhysX collision remains enabled.
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeInvisible()
                matrix = cache.GetLocalToWorldTransform(prim) if prim.IsValid() else None
                translation = list(matrix.ExtractTranslation()) if matrix is not None else None
                # Walk the support prim (and descendants) for any CollisionAPI
                # geometry.  GroundPlaneCfg historically hid its collider under
                # ``{path}/Environment`` while this readback probed a
                # nonexistent ``{path}/geometry/mesh`` and reported false.
                collision_rows: list[dict[str, str]] = []
                if prim.IsValid():
                    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                        if descendant.HasAPI(UsdPhysics.CollisionAPI):
                            collision_rows.append(
                                {
                                    "path": str(descendant.GetPath()),
                                    "collision_enabled": bool(
                                        UsdPhysics.CollisionAPI(descendant)
                                        .GetCollisionEnabledAttr()
                                        .Get()
                                    ),
                                }
                            )
                ray = None
                if translation is not None:
                    result = scene_query.raycast_closest(
                        carb.Float3(float(translation[0]), float(translation[1]), float(translation[2]) + 2.0),
                        carb.Float3(0.0, 0.0, -1.0),
                        5.0,
                    )
                    ray = {
                        "hit": bool(result.get("hit", False)),
                        "position": list(result.get("position", ())) if result.get("hit", False) else None,
                        "collision": str(result.get("collision", "")),
                    }
                readback.append(
                    {
                        "path": path,
                        "valid": bool(prim.IsValid()),
                        "prim_type": prim.GetTypeName() if prim.IsValid() else None,
                        "authored_visibility": (
                            str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get())
                            if prim.IsValid()
                            else None
                        ),
                        "world_translation": translation,
                        "collision_api_prim_paths": [row["path"] for row in collision_rows],
                        "collision_api_count": len(collision_rows),
                        "collision_enabled_all": bool(
                            collision_rows
                            and all(row["collision_enabled"] for row in collision_rows)
                        ),
                        "mesh_path": go2_support_metadata.get("mesh_path", path),
                        "raycast": ray,
                        "support_collision_adopted": bool(
                            collision_rows
                            and all(row["collision_enabled"] for row in collision_rows)
                            and ray is not None
                            and ray["hit"]
                        ),
                    }
                )
            go2_support_metadata["composed_readback"] = readback
            write_json(metadata_dir / "go2_runtime_physics_support.json", go2_support_metadata)
            print("GO2_RUNTIME_SUPPORT_READBACK " + json.dumps(readback, sort_keys=True), flush=True)
        if args.go2_highlight_color is not None:
            go2_highlight_metadata = apply_go2_highlight_material(
                omni.usd.get_context().get_stage(),
                args.go2_highlight_color,
                args.go2_highlight_emission,
            )
            print("GO2_HIGHLIGHT " + json.dumps(go2_highlight_metadata, sort_keys=True), flush=True)
        print("RTX_DIAGNOSTIC phase=sensor_lookup_begin", flush=True)
        if three_camera_definitions is not None:
            from urbanverse.dynamic_agents.rendering.three_camera_writer import ThreeCameraWriter
            three_camera_writer = ThreeCameraWriter(run_dir, three_camera_definitions, unwrapped.scene)
        overview_sensor = (
            unwrapped.scene["go2_traffic_overview"]
            if args.overview_video and not args.roaming_ghost_two_panel_video
            else None
        )
        go2_follow_sensor = (
            unwrapped.scene["go2_traffic_go2_follow"]
            if joint_people_video
            and (not args.roaming_ghost_two_panel_video or project_native_two_panel)
            else None
        )
        person_follow_sensor = (
            unwrapped.scene["go2_traffic_person_follow"]
            if joint_people_video
            and (not args.roaming_ghost_two_panel_video or project_native_two_panel)
            else None
        )
        scene_global_sensor = (
            unwrapped.scene["go2_traffic_scene_global"]
            if args.joint_four_panel_video
            else None
        )
        print(
            "RTX_DIAGNOSTIC phase=sensor_lookup_end "
            f"overview={overview_sensor is not None} "
            f"go2_follow={go2_follow_sensor is not None} "
            f"person_follow={person_follow_sensor is not None} "
            f"scene_global={scene_global_sensor is not None}",
            flush=True,
        )
        manual_two_panel_annotators: dict[str, Any] = {}
        manual_two_panel_products: dict[str, Any] = {}
        manual_two_panel_camera_paths: dict[str, str] = {}
        if args.roaming_ghost_two_panel_video and not project_native_two_panel:
            print("RTX_DIAGNOSTIC phase=manual_products_begin", flush=True)
            import omni.replicator.core as rep
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            manual_camera_specs = [
                (
                    "go2",
                    "/World/TrafficGo2FollowCamera",
                    (args.overview_width, args.overview_height),
                    args.overview_focal_length,
                ),
                (
                    "person",
                    "/World/TrafficPersonFollowCamera",
                    (args.overview_width, args.overview_height),
                    args.overview_focal_length,
                ),
            ]
            if road_sweep_payload is not None:
                sweep_resolution = tuple(
                    map(int, road_sweep_payload["output_resolution"])
                )
                manual_camera_specs.append(
                    (
                        "road_sweep",
                        "/World/TrafficRoadSweepCamera",
                        sweep_resolution,
                        float(road_sweep_payload["focal_length_mm"]),
                    )
                )
            for key, path, resolution, focal_length in manual_camera_specs:
                print(
                    f"RTX_DIAGNOSTIC phase=manual_camera_define_begin key={key}",
                    flush=True,
                )
                camera = UsdGeom.Camera.Define(stage, path)
                camera.CreateFocalLengthAttr(focal_length)
                camera.CreateHorizontalApertureAttr(20.955)
                camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))
                # A manual Replicator camera remains usable while its USD
                # Imageable visibility is disabled.  Hiding it prevents the
                # *other* panel camera from recording this camera's bright
                # green viewport/frustum guide lines.
                UsdGeom.Imageable(camera.GetPrim()).MakeInvisible()
                product = rep.create.render_product(
                    camera.GetPath(),
                    resolution,
                    force_new=True,
                )
                print(
                    f"RTX_DIAGNOSTIC phase=manual_render_product_end key={key}",
                    flush=True,
                )
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([product])
                print(
                    f"RTX_DIAGNOSTIC phase=manual_annotator_attach_end key={key}",
                    flush=True,
                )
                manual_two_panel_camera_paths[key] = path
                manual_two_panel_products[key] = product
                manual_two_panel_annotators[key] = annotator
            print("RTX_DIAGNOSTIC phase=manual_products_end", flush=True)
            # Render products are deliberately created only after the RL
            # environment and official People runtime have reset.  They are
            # initialized on the first scheduled RGB capture below.
        capture_driver_sensor = (
            overview_sensor
            if overview_sensor is not None
            else go2_follow_sensor
            if go2_follow_sensor is not None
            else True
            if manual_two_panel_annotators
            else None
        )
        overview_exposure_readback = None
        overview_motion_blur_readback = None
        if capture_driver_sensor is not None:
            print("RTX_DIAGNOSTIC phase=render_settings_begin", flush=True)
            # AppLauncher and renderer initialization can overwrite tonemapping
            # settings. Apply exposure only after the environment and camera
            # exist, then record the actual readback used by the capture.
            settings = carb.settings.get_settings()
            settings.set("/rtx/post/tonemap/exposure", args.overview_exposure_ev)
            # This overview is geometric interaction evidence.  Explicitly
            # disable temporal blur so a fast vehicle or its shadow cannot be
            # mistaken for a transparent body at the edge of the chase view.
            motion_blur_enabled = bool(args.overview_motion_blur)
            settings.set("/omni/replicator/captureMotionBlur", motion_blur_enabled)
            settings.set("/rtx/post/aa/op", 2)
            settings.set("/rtx/post/motionblur/enabled", motion_blur_enabled)
            settings.set("/rtx/post/motionBlur/enabled", motion_blur_enabled)
            overview_exposure_readback = float(settings.get("/rtx/post/tonemap/exposure"))
            overview_motion_blur_readback = {
                "replicator_capture_motion_blur": bool(
                    settings.get("/omni/replicator/captureMotionBlur")
                ),
                "rtx_aa_operation": int(settings.get("/rtx/post/aa/op")),
                "rtx_motion_blur_enabled": bool(
                    settings.get("/rtx/post/motionblur/enabled")
                ),
                "rtx_motion_blur_camelcase_enabled": bool(
                    settings.get("/rtx/post/motionBlur/enabled")
                ),
            }
            print("RTX_DIAGNOSTIC phase=render_settings_end", flush=True)
        overview_frame_count = 0
        overview_preview_captured = False
        overview_keyframes: list[Path] = []
        road_sweep_frame_count = 0
        road_sweep_keyframes: list[Path] = []
        road_sweep_endpoint_reached = False
        road_sweep_completion_time_s = None
        road_sweep_progress_m = 0.0
        road_sweep_route_length_m = None
        road_sweep_video_path = None
        scene_video_prefix = (
            args.traffic_scene_config.stem
            if args.traffic_scene_config is not None
            else "scene10_go2_multivehicle"
        )
        overview_video_stem = (
            str(road_sweep_payload["output_name"])
            if project_native_road_sweep
            else f"{scene_video_prefix}_four_panel.webm"
            if args.joint_four_panel_video
            else (
                f"{scene_video_prefix}_roaming_ghost_two_panel.webm"
                if ghost_mode_enabled
                else f"{scene_video_prefix}_roaming_two_panel.webm"
            )
            if args.roaming_ghost_two_panel_video
            else f"{scene_video_prefix}_three_panel.webm"
            if args.joint_three_panel_video
            else f"{scene_video_prefix}.webm"
            if args.overview_motion_start_eye is not None
            else f"{scene_video_prefix}_front_pinhole.webm"
            if args.go2_front_pinhole
            else f"{scene_video_prefix}_overview.webm"
        )
        overview_video_path = captures_dir / Path(overview_video_stem).with_suffix(
            ".mp4" if args.overview_video_codec == "h264" else ".webm"
        )
        overview_capture_stride = max(1, int(round(1.0 / (args.overview_fps * dt))))
        actual_overview_fps = 1.0 / (overview_capture_stride * dt)
        if capture_driver_sensor is not None:
            print("RTX_DIAGNOSTIC phase=initial_camera_pose_begin", flush=True)
            import cv2

            initial_robot_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            initial_robot_quaternion = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            if project_native_road_sweep:
                (
                    eye,
                    target,
                    road_sweep_progress_m,
                    road_sweep_route_length_m,
                    road_sweep_endpoint_reached,
                ) = polyline_camera_view(
                    0.0,
                    road_sweep_eye_points,
                    float(road_sweep_payload["speed_mps"]),
                    float(road_sweep_payload["pitch_down_deg"]),
                    float(road_sweep_payload["lookahead_m"]),
                )
            elif args.overview_motion_start_eye is not None:
                eye, target = moving_overview_camera_view(
                    0.0,
                    np.asarray(args.overview_motion_start_eye, dtype=np.float64),
                    np.asarray(args.overview_motion_end_eye, dtype=np.float64),
                    args.overview_motion_speed_mps,
                    args.overview_motion_pitch_down_deg,
                    args.overview_motion_lookahead_m,
                )
            elif args.overview_fixed_eye is not None:
                eye = np.asarray(args.overview_fixed_eye, dtype=np.float64)
                target = np.asarray(args.overview_fixed_target, dtype=np.float64)
            else:
                eye, target = capture.overview_camera_view(
                    initial_robot_position,
                    initial_robot_quaternion,
                    chase_distance=args.overview_chase_distance,
                    lateral_offset=args.overview_lateral_offset,
                    chase_height=args.overview_chase_height,
                    target_forward=args.overview_target_forward,
                    target_height=0.35,
                )
            for follow_sensor in (
                overview_sensor,
                go2_follow_sensor,
                person_follow_sensor,
                scene_global_sensor,
            ):
                if follow_sensor is not None and not args.go2_front_pinhole:
                    follow_sensor.set_world_poses_from_view(
                        torch.tensor(
                            [eye.tolist()], device=unwrapped.device, dtype=torch.float32
                        ),
                        torch.tensor(
                            [target.tolist()], device=unwrapped.device, dtype=torch.float32
                        ),
                    )
            print("RTX_DIAGNOSTIC phase=initial_camera_pose_end", flush=True)
            output_frame_size = (
                args.overview_width
                * (
                    2
                    if args.joint_four_panel_video or args.roaming_ghost_two_panel_video
                    else 3
                    if args.joint_three_panel_video
                    else 1
                ),
                args.overview_height * (2 if args.joint_four_panel_video else 1),
            )
            if args.overview_video_codec == "h264":
                video_writer = FfmpegH264Writer(
                    overview_video_path,
                    actual_overview_fps,
                    output_frame_size,
                    args.overview_h264_crf,
                    args.overview_h264_preset,
                    metadata_dir / "overview_h264_encoder.log",
                )
            else:
                video_writer = cv2.VideoWriter(
                    str(overview_video_path),
                    cv2.VideoWriter_fourcc(*"VP90"),
                    actual_overview_fps,
                    output_frame_size,
                )
            if not video_writer.isOpened():
                raise RuntimeError(f"failed to open overview video writer: {overview_video_path}")
            print("RTX_DIAGNOSTIC phase=overview_writer_ready", flush=True)
            if road_sweep_payload is not None and not project_native_road_sweep:
                road_sweep_width, road_sweep_height = map(
                    int, road_sweep_payload["output_resolution"]
                )
                road_sweep_video_path = captures_dir / str(
                    road_sweep_payload["output_name"]
                )
                road_sweep_video_path = road_sweep_video_path.with_suffix(
                    ".mp4" if args.overview_video_codec == "h264" else ".webm"
                )
                if args.overview_video_codec == "h264":
                    road_sweep_video_writer = FfmpegH264Writer(
                        road_sweep_video_path,
                        actual_overview_fps,
                        (road_sweep_width, road_sweep_height),
                        args.overview_h264_crf,
                        args.overview_h264_preset,
                        metadata_dir / "road_sweep_h264_encoder.log",
                    )
                else:
                    road_sweep_video_writer = cv2.VideoWriter(
                        str(road_sweep_video_path),
                        cv2.VideoWriter_fourcc(*"VP90"),
                        actual_overview_fps,
                        (road_sweep_width, road_sweep_height),
                    )
                if not road_sweep_video_writer.isOpened():
                    raise RuntimeError(
                        f"failed to open road sweep video writer: {road_sweep_video_path}"
                    )
                print("RTX_DIAGNOSTIC phase=road_sweep_writer_ready", flush=True)
            elif project_native_road_sweep:
                road_sweep_video_path = overview_video_path
        print("RTX_DIAGNOSTIC phase=runtime_managers_begin", flush=True)
        traffic_manager = None
        if args.scene10_multivehicle_traffic or args.multivehicle_traffic:
            if args.mode != "route":
                raise ValueError("multivehicle traffic is supported only in route mode")
            portable_scene = (
                TrafficSceneConfig.load(args.traffic_scene_config)
                if args.traffic_scene_config is not None
                else None
            )
            if portable_scene is not None:
                args.traffic_registry = portable_scene.vehicle_catalog
                args.traffic_automotive_routes = portable_scene.automotive_routes
            if args.traffic_registry is None:
                raise ValueError("traffic mode requires --traffic-registry or --traffic-scene-config")
            if args.traffic_audit_inventory is None and portable_scene is None:
                raise ValueError("legacy Scene 10 traffic requires --traffic-audit-inventory")
            manager_class = (
                ContinuousVehicleManager
                if args.traffic_continuous_looping
                else MultiVehicleManager
            )
            if args.traffic_continuous_looping and args.traffic_automotive_routes is None:
                raise ValueError(
                    "continuous traffic requires --traffic-automotive-routes"
                )
            manager_kwargs = {}
            if portable_scene and portable_scene.traffic.get('asset_ids'):
                manager_kwargs['traffic_asset_ids']=portable_scene.traffic['asset_ids']
            if args.traffic_continuous_looping:
                manager_kwargs.update(
                    automotive_routes_path=args.traffic_automotive_routes.resolve(),
                    random_seed=args.seed,
                    traffic_vehicle_count=args.traffic_vehicle_count,
                    initial_fill=(
                        args.traffic_initial_fill
                        or bool(portable_scene.traffic.get("initial_fill", False))
                    ),
                )
            traffic_manager = manager_class(
                omni.usd.get_context().get_stage(),
                omni.physx.get_physx_scene_query_interface(),
                args.traffic_registry.resolve(),
                args.traffic_audit_inventory.resolve() if args.traffic_audit_inventory else None,
                dt=dt,
                validated_routes_path=(
                    args.traffic_validated_routes.resolve() if args.traffic_validated_routes else None
                ),
                validated_static_bodies_path=(
                    args.traffic_validated_static_bodies.resolve()
                    if args.traffic_validated_static_bodies
                    and not args.traffic_continuous_looping
                    else None
                ),
                keep_vehicle_prims_active=args.overview_video,
                opposing_showcase_speed_mps=args.opposing_showcase_speed,
                opposing_showcase_start_route_index=args.opposing_showcase_start_route_index,
                visibility_diagnostic_convoy=args.traffic_visibility_diagnostic_convoy,
                fabric_xform_views=None,
                # Portable vehicles are already one independently controlled
                # prim per agent.  Reusing those references avoids incorrectly
                # treating a USD reference as a GLB payload when RTX cameras
                # request that vehicle prims remain active.
                use_source_vehicle_prims=bool(preauthored_traffic_visuals),
                preauthored_vehicle_prim_paths=[
                    row["controlled_prim"] for row in preauthored_traffic_visuals
                ],
                scene_config_path=(
                    args.traffic_scene_config.resolve() if args.traffic_scene_config else None
                ),
                **manager_kwargs,
            )
        print("RTX_DIAGNOSTIC phase=traffic_manager_ready", flush=True)
        if pedestrian_config is not None:
            pedestrian_config["simulation_dt_s"] = dt
        people_manager = (
            ProjectPeopleRoamingManager(
                omni.usd.get_context().get_stage(),
                pedestrian_config,
                run_dir,
                project_walkable_regions,
                project_roaming_assignments,
                seed=args.seed + 911,
                arrival_radius_m=float(
                    mixed_roaming_payload["people"].get("arrival_radius_m", 0.55)
                ),
                minimum_trip_m=float(
                    mixed_roaming_payload["people"]["minimum_trip_m"]
                ),
                maximum_trip_m=float(
                    mixed_roaming_payload["people"]["maximum_trip_m"]
                ),
                pedestrian_radius_m=float(
                    mixed_roaming_payload["people"].get("radius_m", 0.30)
                ),
                hard_clearance_m=float(
                    mixed_roaming_payload["people"].get("hard_clearance_m", 0.08)
                ),
                stall_retarget_after_s=float(
                    mixed_roaming_payload["people"].get("stall_retarget_after_s", 6.0)
                ),
                path_spacing_m=float(
                    mixed_roaming_payload["people"].get("path_spacing_m", 0.75)
                ),
                raster_admission_tolerance_m=float(
                    mixed_roaming_payload.get("interaction_bands", {}).get(
                        "people_band_tolerance_m", 0.35
                    )
                ),
            )
            if mixed_roaming_payload is not None
            else OfficialPeopleManager(
                omni.usd.get_context().get_stage(), pedestrian_config, run_dir
            )
            if pedestrian_config is not None
            else None
        )
        print("RTX_DIAGNOSTIC phase=people_manager_ready", flush=True)
        official_people_runtime = None
        roaming_people_manager = None
        micromobility_manager = None
        micromobility_usd = None
        micromobility_specs = ()
        micromobility_states = {}
        micro_catalog = {}
        mixed_walkable_audit = None
        approved_navmesh_metadata = None
        roaming_assignments = ()
        latest_official_people_positions = np.empty((0, 3), dtype=np.float64)
        if roaming_ghost_payload is not None:
            stage = omni.usd.get_context().get_stage()
            if (
                stage is None
                or prebaked_official_people_runtime is None
                or prebaked_approved_walkable_union is None
                or prebaked_walkable_regions is None
                or prebaked_micromobility_regions is None
                or prebaked_navmesh_metadata is None
                or prebaked_people_assets_root is None
                or not prebaked_roaming_assignments
                or prebaked_roaming_waypoint_loops is None
                or prebaked_official_setup_metadata is None
            ):
                raise RuntimeError("official NavMesh preload state is incomplete")
            if (
                stage.GetRootLayer().identifier
                != prebaked_official_people_runtime.stage.GetRootLayer().identifier
            ):
                raise RuntimeError("Isaac Lab replaced the pre-baked USD stage")
            stage.SetEditTarget(stage.GetSessionLayer())
            walkable_regions = prebaked_walkable_regions
            micromobility_regions = prebaked_micromobility_regions
            approved_walkable_union = prebaked_approved_walkable_union
            approved_navmesh_metadata = prebaked_navmesh_metadata
            people_payload = roaming_ghost_payload["official_people"]
            people_assets_root = prebaked_people_assets_root
            roaming_assignments = prebaked_roaming_assignments
            roaming_waypoint_loops = prebaked_roaming_waypoint_loops
            official_people_runtime = prebaked_official_people_runtime
            official_navmesh_metadata = approved_navmesh_metadata
            official_setup_metadata = prebaked_official_setup_metadata

            micro_payload = roaming_ghost_payload["micromobility"]
            micromobility_enabled = bool(micro_payload.get("enabled", True))
            micromobility_count = int(micro_payload["count"])
            if micromobility_enabled and micromobility_count > 0:
                micro_catalog = load_micromobility_catalog(
                    resolve_combined(micro_payload["catalog"])
                )
                component_ids = np.asarray(
                    eligible_micromobility_components(
                        micromobility_regions,
                        micro_catalog,
                        minimum_trip_m=float(micro_payload["minimum_trip_m"]),
                    ),
                    dtype=np.int32,
                )
                component_weights = np.asarray(
                    [
                        len(micromobility_regions.component_pixels[int(value)])
                        for value in component_ids
                    ],
                    dtype=np.float64,
                )
                component_weights /= component_weights.sum()
                micro_rng = np.random.default_rng(args.seed + 1701)
                assigned_micro_components = balanced_component_assignment(
                    component_ids,
                    component_weights,
                    count=micromobility_count,
                    rng=micro_rng,
                )
                micro_asset_ids = tuple(micro_catalog)
                micromobility_specs = tuple(
                    MicromobilityAgentSpec(
                        agent_id=f"TwoWheeler_{index:02d}",
                        asset_id=micro_asset_ids[index % len(micro_asset_ids)],
                        component_id=int(component),
                        desired_speed_mps=float(micro_payload["desired_speed_mps"]),
                    )
                    for index, component in enumerate(assigned_micro_components)
                )
                micromobility_manager = MicromobilityRoamingManager(
                    micromobility_regions,
                    micromobility_specs,
                    micro_catalog,
                    seed=args.seed + 2903,
                    minimum_trip_m=float(micro_payload["minimum_trip_m"]),
                    maximum_trip_m=(
                        None
                        if micro_payload.get("maximum_trip_m") is None
                        else float(micro_payload["maximum_trip_m"])
                    ),
                    excluded_spawn_xy=(
                        [item.start_xyz[:2] for item in roaming_assignments]
                        + [item.initial_target_xyz[:2] for item in roaming_assignments]
                        + [
                            point
                            for points in roaming_waypoint_loops.values()
                            for point in points
                        ]
                    ),
                )
                converted_micro = convert_micromobility_assets(
                    simulation_app,
                    micro_catalog,
                    PROJECT_ROOT / "data",
                    run_dir / "converted_micromobility",
                )
                micromobility_usd = MicromobilityUsdRuntime(
                    stage,
                    micromobility_specs,
                    micro_catalog,
                    converted_micro,
                    ground_z_m=float(
                        micro_payload.get(
                            "visual_support_z_m", walkable_regions.config.ground_z_m
                        )
                    ),
                    scene_query=(
                        omni.physx.get_physx_scene_query_interface()
                        if bool(micro_payload.get("runtime_support_raycast", False))
                        else None
                    ),
                )
                micromobility_usd.update(micromobility_manager.states)
            roaming_people_manager = OfficialPeopleRoamingManager(
                official_people_runtime,
                walkable_regions,
                roaming_assignments,
                seed=args.seed + 911,
                arrival_radius_m=float(people_payload["arrival_radius_m"]),
                minimum_trip_m=float(people_payload["minimum_trip_m"]),
                maximum_trip_m=float(people_payload.get("maximum_target_leg_m", 30.0)),
                waypoint_loops=roaming_waypoint_loops,
                reassign_on_arrival=True,
                dynamic_target_clearance_m=float(
                    people_payload.get("dynamic_target_clearance_m", 3.0)
                ),
                dynamic_proximity_retarget_m=float(
                    people_payload.get("dynamic_proximity_retarget_m", 2.5)
                ),
                people_target_clearance_m=float(
                    people_payload.get("people_target_clearance_m", 1.5)
                ),
                people_proximity_retarget_m=float(
                    people_payload.get("people_proximity_retarget_m", 2.2)
                ),
                people_yield_duration_s=float(
                    people_payload.get("people_yield_duration_s", 2.5)
                ),
                stall_retarget_after_s=float(
                    people_payload.get("stall_retarget_after_s", 10.0)
                ),
            )
            mixed_walkable_audit = MixedWalkableAgentAudit(
                walkable_regions,
                roaming_assignments,
                micromobility_specs,
                micro_catalog,
                micromobility_regions=micromobility_regions,
                approved_union_regions=approved_walkable_union,
                people_raster_tolerance_m=float(
                    roaming_ghost_payload.get("interaction_bands", {}).get(
                        "people_band_tolerance_m", 0.35
                    )
                ),
            )
            # CharacterManager was fully initialized before Isaac Lab created
            # its PhysX tensor views.  Reuse those handles without any Kit
            # update here.  Isaac Lab may have paused the timeline while
            # constructing the environment; resuming it does not mutate the
            # stage or rebuild physics objects.
            official_play_updates = prebaked_official_play_updates
            if official_people_runtime.timeline is not None and not official_people_runtime.timeline.is_playing():
                official_people_runtime.timeline.play()
                official_people_runtime.timeline.commit()
            # Isaac Lab's environment reset preserves the already-created
            # CharacterManager handles but clears the command status that was
            # read from the initial command file.  Re-issue the first validated
            # target through NVIDIA's official runtime injection API.  This is
            # a command/state operation only: it performs no Kit update and
            # therefore cannot invalidate the new PhysX tensor views.
            for assignment in roaming_assignments:
                official_people_runtime.inject_goto(
                    assignment.name, assignment.initial_target_xyz
                )
            latest_official_people_positions = official_people_runtime.positions()
            write_json(
                metadata_dir / "official_roaming_setup.json",
                {
                    "extensions": official_extensions,
                    "navmesh": approved_navmesh_metadata,
                    "official_setup": official_setup_metadata,
                    "official_play_updates": official_play_updates,
                    "walkable_regions": walkable_regions.summary(),
                    "approved_walkable_union": approved_walkable_union.summary(),
                    "micromobility_regions": micromobility_regions.summary(),
                    "interaction_bands": roaming_ghost_payload.get(
                        "interaction_bands"
                    ),
                    "resident_assignments": [
                        {
                            "name": item.name,
                            "asset": str(item.asset),
                            "component_id": item.component_id,
                            "start_xyz": list(item.start_xyz),
                            "initial_target_xyz": list(item.initial_target_xyz),
                        }
                        for item in roaming_assignments
                    ],
                    "resident_target_cycles_xy": {
                        name: points.tolist()
                        for name, points in roaming_waypoint_loops.items()
                    },
                    "resident_logical_target_cycles_xy": {
                        name: points.tolist()
                        for name, points in prebaked_roaming_logical_waypoint_loops.items()
                    },
                    "micromobility_specs": [
                        {
                            "agent_id": item.agent_id,
                            "asset_id": item.asset_id,
                            "component_id": item.component_id,
                            "desired_speed_mps": item.desired_speed_mps,
                        }
                        for item in micromobility_specs
                    ],
                    "official_dynamic_obstacle_scripts": (
                        micromobility_usd.dynamic_obstacle_scripts
                        if micromobility_usd is not None
                        else []
                    ),
                },
            )
            # No Go2 root pose is written by this integration.  The calls above
            # are the standard Isaac Lab simulation/environment reset pair.
        elif mixed_roaming_payload is not None:
            if (
                project_approved_walkable_union is None
                or project_walkable_regions is None
                or project_micromobility_regions is None
                or people_manager is None
                or not project_roaming_assignments
            ):
                raise RuntimeError("project mixed-roaming preload state is incomplete")
            stage = omni.usd.get_context().get_stage()
            stage.SetEditTarget(stage.GetSessionLayer())
            approved_walkable_union = project_approved_walkable_union
            walkable_regions = project_walkable_regions
            micromobility_regions = project_micromobility_regions
            roaming_assignments = project_roaming_assignments
            people_payload = mixed_roaming_payload["people"]
            micro_payload = mixed_roaming_payload["micromobility"]
            micromobility_enabled = bool(micro_payload.get("enabled", True))
            micromobility_count = int(micro_payload["count"])
            if micromobility_enabled and micromobility_count > 0:
                print("RTX_DIAGNOSTIC phase=project_micro_planning_begin", flush=True)
                micro_catalog = load_micromobility_catalog(
                    resolve_combined(micro_payload["catalog"])
                )
                project_micro_loops={}
                if closed_loop_document is not None:
                    project_micro_loops=distribute(closed_loop_document['groups']['micromobility']['routes'],
                        micromobility_count,'TwoWheeler',micromobility_regions,
                        bool(mixed_roaming_payload.get('reverse_alternate_loops',False)),
                        reduced_allocation(closed_loop_document.get('population_proposal',{}).get('micromobility',{}).get('per_route'),int(mixed_roaming_payload['micromobility']['count'])),
                        2.5,
                        [item.start_xyz[:2] for item in roaming_assignments])
                component_ids = np.asarray(
                    sorted({item['component_id'] for item in project_micro_loops.values()}) if project_micro_loops else eligible_micromobility_components(
                        micromobility_regions,
                        micro_catalog,
                        minimum_trip_m=float(micro_payload["minimum_trip_m"]),
                    ),
                    dtype=np.int32,
                )
                component_weights = np.asarray(
                    [
                        len(micromobility_regions.component_pixels[int(value)])
                        for value in component_ids
                    ],
                    dtype=np.float64,
                )
                component_weights /= component_weights.sum()
                assigned_micro_components = balanced_component_assignment(
                    component_ids,
                    component_weights,
                    count=micromobility_count,
                    rng=np.random.default_rng(args.seed + 1701),
                )
                if project_micro_loops:
                    assigned_micro_components=[item['component_id'] for item in project_micro_loops.values()]
                micro_asset_ids = tuple(micro_catalog)
                micromobility_specs = tuple(
                    MicromobilityAgentSpec(
                        agent_id=f"TwoWheeler_{index:02d}",
                        asset_id=micro_asset_ids[index % len(micro_asset_ids)],
                        component_id=int(component),
                        desired_speed_mps=float(micro_payload["desired_speed_mps"]),
                    )
                    for index, component in enumerate(assigned_micro_components)
                )
                micromobility_manager = MicromobilityRoamingManager(
                    micromobility_regions,
                    micromobility_specs,
                    micro_catalog,
                    seed=args.seed + 2903,
                    minimum_trip_m=float(micro_payload["minimum_trip_m"]),
                    maximum_trip_m=float(micro_payload["maximum_trip_m"]),
                    excluded_spawn_xy=[item.start_xyz[:2] for item in roaming_assignments],
                    closed_paths_xy={name:item['xy'] for name,item in project_micro_loops.items()},
                )
                print("RTX_DIAGNOSTIC phase=project_micro_planning_end", flush=True)
                print("RTX_DIAGNOSTIC phase=project_micro_conversion_begin", flush=True)
                converted_micro = preconverted_project_micro if preconverted_project_micro is not None else convert_micromobility_assets(
                    simulation_app,
                    micro_catalog,
                    PROJECT_ROOT / "data",
                    run_dir / "converted_micromobility",
                )
                print("RTX_DIAGNOSTIC phase=project_micro_conversion_end", flush=True)
                print("RTX_DIAGNOSTIC phase=project_micro_usd_begin", flush=True)
                micromobility_usd = MicromobilityUsdRuntime(
                    stage,
                    micromobility_specs,
                    micro_catalog,
                    converted_micro,
                    ground_z_m=float(
                        micro_payload.get(
                            "visual_support_z_m", walkable_regions.config.ground_z_m
                        )
                    ),
                    register_official_dynamic_obstacles=False,
                    scene_query=(
                        omni.physx.get_physx_scene_query_interface()
                        if bool(micro_payload.get("runtime_support_raycast", False))
                        else None
                    ),
                )
                micromobility_usd.update(micromobility_manager.states)
                print("RTX_DIAGNOSTIC phase=project_micro_usd_end", flush=True)
            mixed_walkable_audit = MixedWalkableAgentAudit(
                walkable_regions,
                roaming_assignments,
                micromobility_specs,
                micro_catalog,
                micromobility_regions=micromobility_regions,
                approved_union_regions=approved_walkable_union,
                people_raster_tolerance_m=float(
                    mixed_roaming_payload.get("interaction_bands", {}).get(
                        "people_band_tolerance_m", 0.35
                    )
                ),
                people_radius_m=float(people_payload.get("radius_m", 0.30)),
            )
            latest_official_people_positions = people_manager.positions()
            write_json(
                metadata_dir / "project_mixed_roaming_setup.json",
                {
                    "control_backend": "project",
                    "character_manager": False,
                    "navmesh": False,
                    "navigation_manager": False,
                    "official_dynamic_obstacles": False,
                    "walkable_regions": walkable_regions.summary(),
                    "approved_walkable_union": approved_walkable_union.summary(),
                    "interaction_bands": mixed_roaming_payload.get("interaction_bands"),
                    "resident_assignments": [
                        {
                            "name": item.name,
                            "asset": str(item.asset),
                            "component_id": item.component_id,
                            "start_xyz": list(item.start_xyz),
                            "initial_target_xyz": list(item.initial_target_xyz),
                        }
                        for item in roaming_assignments
                    ],
                    "micromobility_specs": [
                        {
                            "agent_id": item.agent_id,
                            "asset_id": item.asset_id,
                            "component_id": item.component_id,
                            "desired_speed_mps": item.desired_speed_mps,
                        }
                        for item in micromobility_specs
                    ],
                },
            )
        print("RTX_DIAGNOSTIC phase=mixed_runtime_ready", flush=True)
        if route_payload.get("live_route_preflight", {}).get("enabled"):
            from urbanverse.dynamic_agents.audit.go2_route_corridor import audit_route_corridor, robot_collision_footprint
            if args.route_review_only:
                from urbanverse.dynamic_agents.audit.go2_route_corridor import audit_lower_row_grid
                write_json(metadata_dir / "live_lower_row_grid.json", audit_lower_row_grid(metadata_dir, route_payload["ground_z"]))
            footprint = robot_collision_footprint(robot)
            write_json(metadata_dir / "robot_collision_footprint.json", footprint)
            if footprint['width_m'] > .55:
                raise RuntimeError(f"Robot nominal collision width exceeds narrow-route admission width: {footprint}")
            preflight = audit_route_corridor(route_payload)
            write_json(metadata_dir / "live_route_corridor.json", preflight)
            print("LIVE_ROUTE_PREFLIGHT " + json.dumps({k:v for k,v in preflight.items() if k!='samples'}), flush=True)
            if capture_driver_sensor is not None:
                for _ in range(3): unwrapped.sim.render()
                rgb = capture_driver_sensor.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
                Image.fromarray(rgb.astype(np.uint8)).save(captures_dir / "route_review_scene.png")
            if args.route_review_only or not preflight["passed"]:
                write_json(metadata_dir / "summary.json", {"status":"review_passed" if preflight["passed"] else "review_failed", "preflight":{k:v for k,v in preflight.items() if k!='samples'}, "motion_started":False})
                write_json(metadata_dir / "environment.json", {"hostname":platform.node(),"git_commit":run_text(["git","rev-parse","HEAD"]),
                    "command_line":sys.argv,"timestamp":datetime.now().astimezone().isoformat(),
                    "gpu":capture.gpu_snapshot(args.gpu),"python_version":sys.version,
                    "kit_version":omni.kit.app.get_app().get_build_version(),
                    "replicator_extension":omni.kit.app.get_app().get_extension_manager().get_enabled_extension_id("omni.replicator.core"),
                    "isaac_sim_version":package_version("isaacsim"),"source_usd":str(args.source_usd),
                    "source_usd_sha256":capture.sha256(args.source_usd),"source_tar_sha256":capture.sha256(args.source_tar) if args.source_tar else None,
                    "gpu_index":args.gpu,"preflight_environment":"see metadata/preflight", "renderer":"RayTracedLighting",
                    "resolution":[args.overview_width,args.overview_height],"camera":str(capture_driver_sensor.cfg.prim_path) if capture_driver_sensor else None})
                if capture_driver_sensor is not None:
                    Image.open(captures_dir / "route_review_scene.png").save(visualizations_dir / "route_review_scene.png")
                return 0 if preflight["passed"] else 1
        if replay_commands is not None and abs(dt - 0.02) > 1.0e-9:
            raise RuntimeError(f"replay source uses 0.02 s commands but environment step is {dt}")
        actor_metadata = None
        external=None
        if args.policy_kind in ("robot_lab", "himloco"):
            from urbanverse.dynamic_agents.navigation.external_go2_policy import ExternalGo2Policy
            external = ExternalGo2Policy(args.external_policy_source, args.policy_kind, robot.joint_names, unwrapped.device, model_path=args.policy.resolve())
            actor_metadata = {"upstream_config":external.cfg,"policy_joint_names":external.policy_names,"sim_joint_names":robot.joint_names,
                "policy_to_sim_indices":external.indices,"source_manifest":json.loads((args.external_policy_source/"manifest.json").read_text())}
            def run_policy(observation):
                return external.act(robot, unwrapped.command_manager.get_term("base_velocity").vel_command_b)[0]
        elif args.policy_kind == "go2z1_v2_checkpoint":
            policy, actor_metadata = build_actor(torch, args.checkpoint.resolve(), f"cuda:{args.gpu}")

            def run_policy(observation):
                return policy(adapt_pure_go2_observation(torch, observation))

        else:
            policy = torch.jit.load(str(args.policy.resolve()), map_location=f"cuda:{args.gpu}").eval()

            def run_policy(observation):
                return policy(observation)
        command_term = unwrapped.command_manager.get_term("base_velocity")
        command_term.time_left[:] = 1000.0
        command_term.is_heading_env[:] = False
        command_term.is_standing_env[:] = False
        print("RTX_DIAGNOSTIC phase=policy_warmup_begin", flush=True)
        if dynamic_contact_metadata is not None:
            from urbanverse.dynamic_agents.integration.dynamic_contact import Runtime as ContactRuntime
            dynamic_contact_runtime=ContactRuntime(omni.usd.get_context().get_stage(),dynamic_contact_metadata)
            dynamic_contact_runtime.update(traffic_manager.agents,people_manager.positions(),micromobility_manager.states)
        with torch.inference_mode():
            for _ in range(int(round(1.0 / dt))):
                command_term.vel_command_b.zero_()
                observation_tensor = observations["policy"] if isinstance(observations, dict) else observations
                actions = run_policy(observation_tensor)
                observations, _, _, _, _ = env.step(actions)
        print("RTX_DIAGNOSTIC phase=policy_warmup_end", flush=True)
        if dynamic_contact_runtime is not None:
            contact_query_audit=dynamic_contact_runtime.audit_physx(omni.physx.get_physx_scene_query_interface())
            write_json(metadata_dir/'dynamic_contact_physx_query.json',contact_query_audit)
            if not contact_query_audit['passed']:
                raise RuntimeError('Dynamic contact proxies not present in PhysX at actor positions')

        controller = None
        if args.mode == "route":
            controller = navigation.PurePursuitController(
                waypoints_world_xy=route_points.tolist(),
                max_forward_speed=args.max_forward_speed,
                max_tracking_yaw_rate=args.max_tracking_yaw_rate,
                minimum_tracking_speed=args.minimum_tracking_speed,
                lookahead_distance=args.lookahead_distance,
                curvature_speed_gain=args.curvature_speed_gain,
                vx_rate_limit=1.2,
                yaw_rate_limit=0.8,
                goal_tolerance=0.35,
                deceleration_distance=0.80,
                settle_duration_s=1.0,
                stop_at_final_waypoint=True,
            )
        approach_controller = None
        exit_controller = None
        maneuver_phase = "approach"
        maneuver_index = 0
        maneuver_phase_started_s = 0.0
        if args.mode == "three_point":
            controller_kwargs = dict(
                max_forward_speed=0.34,
                max_tracking_yaw_rate=0.18,
                minimum_tracking_speed=args.minimum_tracking_speed,
                lookahead_distance=args.lookahead_distance,
                curvature_speed_gain=2.0,
                vx_rate_limit=1.2,
                yaw_rate_limit=0.8,
                goal_tolerance=0.35,
                deceleration_distance=0.80,
                settle_duration_s=0.5,
                stop_at_final_waypoint=True,
            )
            approach_controller = navigation.PurePursuitController(
                waypoints_world_xy=maneuver_payload["approach_points_xy"], **controller_kwargs
            )
            exit_controller = navigation.PurePursuitController(
                waypoints_world_xy=maneuver_payload["exit_points_xy"], **controller_kwargs
            )
        stuck_monitor = navigation.StuckMonitor(
            window_s=10.0,
            command_speed_threshold=0.15,
            minimum_progress=0.08,
            command_yaw_rate_threshold=0.10,
            minimum_yaw_progress=0.10,
            confirmation_s=20.0,
        )
        collision_monitor = navigation.SustainedCollisionMonitor(duration_s=0.5)
        recovery=None
        if dynamic_contact_runtime is not None:
            from urbanverse.dynamic_agents.integration.forward_recovery import ForwardRecovery,dynamic_circles
            from urbanverse.dynamic_agents.navigation.mesh_runtime import from_scene
            recovery_road=from_scene(TrafficSceneConfig.load(args.traffic_scene_config))
            recovery=ForwardRecovery(route_points,route_arc,float(route_payload['ground_z']),
                recovery_road.grid,recovery_road.free,mixed_roaming_payload['go2']['collision_recovery'])
        trajectory_path = metadata_dir / "trajectory.jsonl"
        trajectory_handle = trajectory_path.open("w", encoding="utf-8")
        if args.mixed_roaming_config is not None and args.traffic_scene_config is not None:
            from urbanverse.dynamic_agents.core.overlap_labels import DynamicOverlapLabeler, OverlapRecorder, rotation
            overlap_labeler = DynamicOverlapLabeler.from_configs(
                args.traffic_scene_config, args.mixed_roaming_config,
                [dict(agent_id=s.agent_id, asset_id=s.asset_id) for s in micromobility_specs])
            overlap_recorder = OverlapRecorder(metadata_dir)
        actual_positions = []
        actual_segments=[]
        commands = []
        body_linear_samples = []
        body_angular_samples = []
        maximum_cross_track = 0.0
        maximum_tracking_curvature = 0.0
        nonfoot_steps = 0
        head_contact_steps = 0
        foot_support_steps = 0
        route_stuck = False
        route_diverged = False
        sustained_collision = False
        goal_reached = False
        goal_reached_at_s = None
        stop_reason = "duration_complete"
        previous_official_people_positions = latest_official_people_positions.copy()
        initial_official_people_positions = latest_official_people_positions.copy()
        maximum_official_people_displacement_m = 0.0
        official_people_runtime_samples: list[dict[str, Any]] = []
        mixed_avoidance_audit: dict[str, Any] = {}
        invalid_intervals: list[dict[str, Any]] = []
        open_invalid_labels: tuple[str, ...] = ()
        open_invalid_start_s: float | None = None
        total_steps = int(round(float(duration_s) / dt))
        run_started = time.perf_counter()
        print("RTX_DIAGNOSTIC phase=simulation_loop_begin", flush=True)
        for step_index in range(total_steps):
            timestamp_before = step_index * dt
            position_before = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            quaternion_before = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            shared_mixed_states = ()
            shared_mixed_speed_scales: dict[str, float] = {}
            traffic_yield_obstacles: list[dict[str, Any]] = []
            if mixed_roaming_payload is not None and people_manager is not None:
                shared_mixed_states = (
                    *people_manager.mixed_states(),
                    *(
                        micromobility_manager.mixed_states()
                        if micromobility_manager is not None
                        else ()
                    ),
                )
                shared_mixed_speed_scales = reciprocal_speed_scales(
                    shared_mixed_states,
                    prediction_horizon_s=float(
                        mixed_roaming_payload.get("avoidance", {}).get(
                            "prediction_horizon_s", 1.5
                        )
                    ),
                    minimum_clearance_m=float(
                        mixed_roaming_payload.get("avoidance", {}).get(
                            "minimum_clearance_m", 0.15
                        )
                    ),
                    slow_clearance_m=float(
                        mixed_roaming_payload.get("avoidance", {}).get(
                            "slow_clearance_m", 1.5
                        )
                    ),
                    audit=mixed_avoidance_audit,
                )
                traffic_yield_obstacles = [
                    {
                        "position_xy": item.position_xy,
                        "radius_m": item.radius_m,
                        "id": item.agent_id,
                        "kind": item.kind,
                    }
                    for item in shared_mixed_states
                ]
                if not dynamic_agents_ignore_go2:
                    traffic_yield_obstacles.append(
                        {
                            "position_xy": position_before[:2],
                            "radius_m": 0.67,
                            "id": "Go2",
                            "kind": "go2",
                        }
                    )
            traffic_state = (
                traffic_manager.update(
                    timestamp_before,
                    (
                        np.asarray((-1.0e6, -1.0e6), dtype=np.float64)
                        if roaming_ghost_payload is not None
                        else position_before[:2]
                    ),
                    capture.yaw_from_quat(quaternion_before),
                    **(
                        {"yield_obstacles": traffic_yield_obstacles}
                        if mixed_roaming_payload is not None
                        and getattr(traffic_manager, "continuous_looping", False)
                        else {}
                    ),
                )
                if traffic_manager is not None
                else None
            )
            if people_manager is not None:
                people_kwargs = (
                    {
                        "micromobility_states": micromobility_states,
                        "micromobility_specs": micromobility_specs,
                        "micromobility_catalog": micro_catalog,
                        "shared_speed_scales": shared_mixed_speed_scales,
                        "ignore_go2": dynamic_agents_ignore_go2,
                    }
                    if mixed_roaming_payload is not None
                    else {}
                )
                people_manager.prepare_step(
                    traffic_manager.agents if traffic_manager is not None else None,
                    position_before[:2],
                    robot.data.root_lin_vel_w[0, :2].detach().cpu().numpy().astype(np.float64),
                    **people_kwargs,
                )
                if mixed_roaming_payload is not None:
                    previous_official_people_positions = latest_official_people_positions.copy()
                    latest_official_people_positions = people_manager.positions()
            if micromobility_manager is not None:
                micromobility_states = micromobility_manager.update(
                    dt,
                    latest_official_people_positions,
                    shared_speed_scales=shared_mixed_speed_scales,
                    traffic_agents=(
                        traffic_manager.agents if traffic_manager is not None else ()
                    ),
                    go2_xy=(
                        None
                        if dynamic_agents_ignore_go2
                        else position_before[:2]
                    ),
                )
                micromobility_usd.update(micromobility_states)
            if dynamic_contact_runtime is not None:
                dynamic_contact_runtime.update(traffic_manager.agents,people_manager.positions(),micromobility_states)
            if args.mode == "constant":
                command = {
                    "label": "constant",
                    "linear_x": args.constant_vx,
                    "linear_y": 0.0,
                    "angular_z": args.constant_wz,
                }
            elif args.mode == "replay":
                replay = replay_commands[min(step_index, len(replay_commands) - 1)]
                command = {"label": "replay", "linear_x": replay[0], "linear_y": replay[1], "angular_z": replay[2]}
            elif args.mode == "route":
                command = controller.command(position_before[:2], capture.yaw_from_quat(quaternion_before), timestamp_before)
                if args.yaw_command_gain != 1.0:
                    raw_yaw_rate = float(command["angular_z"])
                    command["angular_z"] = float(
                        np.clip(
                            raw_yaw_rate * args.yaw_command_gain,
                            -args.max_tracking_yaw_rate,
                            args.max_tracking_yaw_rate,
                        )
                    )
                    command.setdefault("tracking_debug", {})["yaw_rate_before_gain_radps"] = raw_yaw_rate
                    command["tracking_debug"]["yaw_command_gain"] = args.yaw_command_gain
            else:
                yaw_before = capture.yaw_from_quat(quaternion_before)
                if maneuver_phase == "approach":
                    command = approach_controller.command(position_before[:2], yaw_before, timestamp_before)
                    if approach_controller.route_complete:
                        maneuver_phase = "maneuver"
                        maneuver_index = 0
                        maneuver_phase_started_s = timestamp_before
                elif maneuver_phase == "maneuver":
                    phases = maneuver_payload["phases"]
                    while maneuver_index < len(phases) and timestamp_before - maneuver_phase_started_s >= float(phases[maneuver_index]["duration_s"]):
                        maneuver_phase_started_s += float(phases[maneuver_index]["duration_s"])
                        maneuver_index += 1
                    if maneuver_index >= len(phases):
                        maneuver_phase = "exit"
                        command = exit_controller.command(position_before[:2], yaw_before, timestamp_before)
                    else:
                        phase = phases[maneuver_index]
                        command = {
                            "label": f"three_point/{phase['label']}",
                            "linear_x": float(phase["linear_x"]),
                            "linear_y": 0.0,
                            "angular_z": float(phase["angular_z"]),
                            "maneuver_phase_index": maneuver_index,
                        }
                else:
                    command = exit_controller.command(position_before[:2], yaw_before, timestamp_before)
            if recovery is not None and recovery.waiting_since is not None:
                command.update(linear_x=0.,linear_y=0.,angular_z=0.,label='waiting_for_safe_forward_recovery')
            command_tensor = torch.tensor(
                [[command["linear_x"], command["linear_y"], command["angular_z"]]],
                dtype=torch.float32,
                device=unwrapped.device,
            )
            command_term.vel_command_b[:] = command_tensor
            observation_tensor = observations["policy"] if isinstance(observations, dict) else observations
            with torch.inference_mode():
                actions = run_policy(observation_tensor)
                observations, rewards, terminated, truncated, _ = env.step(actions)
            if roaming_people_manager is not None and (
                not args.overview_video or manual_two_panel_annotators
            ):
                # Tensor-only Isaac Lab does not emit Kit's stage-update event,
                # while SimulationApp.update() would advance PhysX a second
                # time.  Tick NVIDIA's already-created CharacterBehavior and
                # NavigationManager instances directly at the controlled dt;
                # the next env.step evaluates their Animation Graph variables.
                official_people_runtime.advance_behaviors(dt)
            if people_manager is not None:
                people_manager.sample()
                if mixed_roaming_payload is not None:
                    maximum_official_people_displacement_m = max(
                        maximum_official_people_displacement_m,
                        float(
                            np.linalg.norm(
                                latest_official_people_positions[:, :2]
                                - initial_official_people_positions[:, :2],
                                axis=1,
                            ).max()
                        ),
                    )
                    if mixed_walkable_audit is not None:
                        mixed_walkable_audit.observe(
                            latest_official_people_positions,
                            micromobility_states,
                            simulation_time_s=(step_index + 1) * dt,
                        )
            if roaming_people_manager is not None:
                previous_official_people_positions = latest_official_people_positions.copy()
                latest_official_people_positions = roaming_people_manager.update(
                    np.asarray(
                        [state.position_xy for state in micromobility_states.values()],
                        dtype=np.float64,
                    )
                    if micromobility_states
                    else np.empty((0, 2), dtype=np.float64),
                    delta_time_s=dt,
                )
                maximum_official_people_displacement_m = max(
                    maximum_official_people_displacement_m,
                    float(
                        np.linalg.norm(
                            latest_official_people_positions[:, :2]
                            - initial_official_people_positions[:, :2],
                            axis=1,
                        ).max()
                    ),
                )
                sample_period_steps = max(1, int(round(5.0 / dt)))
                if step_index % sample_period_steps == 0:
                    official_people_runtime_samples.append(
                        {
                            "simulation_time_s": (step_index + 1) * dt,
                            "positions_xy": latest_official_people_positions[:, :2].tolist(),
                            "maximum_displacement_from_start_m": (
                                maximum_official_people_displacement_m
                            ),
                            "official_timeline_time_s": float(
                                official_people_runtime.timeline.get_current_time()
                            ),
                        }
                    )
                if mixed_walkable_audit is not None:
                    mixed_walkable_audit.observe(
                        latest_official_people_positions,
                        micromobility_states,
                        simulation_time_s=(step_index + 1) * dt,
                    )
            timestamp = (step_index + 1) * dt
            position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            quaternion = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            yaw = capture.yaw_from_quat(quaternion)
            velocity_body = robot.data.root_lin_vel_b[0].detach().cpu().numpy().astype(np.float64)
            angular_body = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float64)
            contact = contact_sensor.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
            contact_norms = np.linalg.norm(contact, axis=-1)
            contact_state = navigation.classify_contacts(contact_sensor.body_names, contact_norms)
            nonfoot = bool(contact_state["non_foot_collision"])
            nonfoot_steps += int(nonfoot)
            foot_support_steps += int(bool(contact_state["supporting_feet"]))
            head_contact_steps += int(any(item["body"].startswith("Head_") for item in contact_state["non_foot_contacts"]))
            collision_state = collision_monitor.update(timestamp, nonfoot)
            stuck_state = stuck_monitor.update(
                timestamp, position[:2], float(command["linear_x"]), yaw, float(command["angular_z"])
            )
            arc_progress, cross_track = closest_route_metrics(position[:2], route_points, route_arc)
            maximum_cross_track = max(maximum_cross_track, cross_track)
            debug = command.get("tracking_debug", {})
            maximum_tracking_curvature = max(maximum_tracking_curvature, abs(float(debug.get("curvature_rad_per_m", 0.0))))
            actual_positions.append(position.tolist())
            actual_segments.append(recovery.segment if recovery is not None else 0)
            commands.append([float(command["linear_x"]), float(command["linear_y"]), float(command["angular_z"])])
            body_linear_samples.append(velocity_body.tolist())
            body_angular_samples.append(angular_body.tolist())
            if overlap_labeler is not None:
                overlap_input = dict(timestamp_s=timestamp, step_index=step_index+1,
                    base_position_world=position.tolist(), base_quaternion_wxyz=quaternion.tolist(),
                    base_yaw_rad_world=yaw, motion_segment_id=recovery.segment if recovery is not None else 0,
                    scene10_multivehicle_traffic=traffic_state,
                    pedestrian_positions_xyz=latest_official_people_positions.tolist(),
                    pedestrian_ids=[item.name for item in roaming_assignments],
                    micromobility_states={name: dict(position_xy=s.position_xy.tolist(), yaw_rad=float(s.yaw_rad))
                                          for name, s in micromobility_states.items()},
                    contact_classification=contact_state)
                overlap_camera = position + rotation(quaternion) @ np.array([.35, 0., .10]) if args.go2_front_pinhole else None
                overlap_record = overlap_labeler.evaluate(overlap_input, overlap_camera,
                    'rigid_mount_full_root_pose' if args.go2_front_pinhole else 'not_available')
                overlap_recorder.observe(overlap_record)
            if ghost_mode_enabled:
                overlap = tuple(
                    sorted(
                        ghost_overlap_labels(
                            position[:2],
                            ghost_experiment,
                            latest_official_people_positions,
                            micromobility_manager.states if micromobility_manager else None,
                            traffic_state,
                        )
                    )
                )
                if overlap != open_invalid_labels:
                    if open_invalid_labels and open_invalid_start_s is not None:
                        invalid_intervals.append(
                            {
                                "start_s": open_invalid_start_s,
                                "end_s": timestamp,
                                "labels": list(open_invalid_labels),
                                "ordinary_navigation_training_valid": False,
                            }
                        )
                    open_invalid_labels = overlap
                    open_invalid_start_s = timestamp if overlap else None
            scheduled_overview_capture = (
                args.overview_preview_time_s is None
                and (step_index % overview_capture_stride == 0 or step_index == total_steps - 1)
            )
            preview_overview_capture = (
                args.overview_preview_time_s is not None
                and not overview_preview_captured
                and timestamp + 0.5 * dt >= args.overview_preview_time_s
            )
            if capture_driver_sensor is not None and (
                scheduled_overview_capture or preview_overview_capture
            ):
                import cv2

                if traffic_manager is not None:
                    traffic_manager.refresh_visual_transforms()
                if people_manager is not None:
                    people_manager.refresh_visual_transforms()

                road_sweep_reached_this_frame = road_sweep_endpoint_reached
                if project_native_road_sweep:
                    (
                        eye,
                        target,
                        road_sweep_progress_m,
                        road_sweep_route_length_m,
                        road_sweep_reached_this_frame,
                    ) = polyline_camera_view(
                        timestamp,
                        road_sweep_eye_points,
                        float(road_sweep_payload["speed_mps"]),
                        float(road_sweep_payload["pitch_down_deg"]),
                        float(road_sweep_payload["lookahead_m"]),
                    )
                elif args.overview_motion_start_eye is not None:
                    eye, target = moving_overview_camera_view(
                        timestamp,
                        np.asarray(
                            args.overview_motion_start_eye, dtype=np.float64
                        ),
                        np.asarray(args.overview_motion_end_eye, dtype=np.float64),
                        args.overview_motion_speed_mps,
                        args.overview_motion_pitch_down_deg,
                        args.overview_motion_lookahead_m,
                    )
                elif args.overview_fixed_eye is not None:
                    eye = np.asarray(args.overview_fixed_eye, dtype=np.float64)
                    target = np.asarray(args.overview_fixed_target, dtype=np.float64)
                elif args.traffic_visibility_diagnostic_convoy and traffic_state is not None:
                    moving_centers = np.asarray(
                        [
                            agent["center_xyz"]
                            for agent in traffic_state["agents"]
                            if agent["active_on_road"]
                        ],
                        dtype=np.float64,
                    )
                    if not len(moving_centers):
                        raise RuntimeError("visibility diagnostic convoy has no active vehicles")
                    target = np.asarray(
                        [moving_centers[:, 0].mean(), moving_centers[:, 1].mean(), 0.35],
                        dtype=np.float64,
                    )
                    eye = target + np.asarray([-10.0, -14.0, 32.0], dtype=np.float64)
                else:
                    eye, target = capture.overview_camera_view(
                        position,
                        quaternion,
                        chase_distance=args.overview_chase_distance,
                        lateral_offset=args.overview_lateral_offset,
                        chase_height=args.overview_chase_height,
                        target_forward=args.overview_target_forward,
                        target_lateral=args.overview_target_lateral,
                        target_height=0.35,
                    )
                def set_sensor_view(sensor: Any, view_eye: np.ndarray, view_target: np.ndarray) -> None:
                    sensor.set_world_poses_from_view(
                        torch.tensor(
                            [view_eye.tolist()], device=unwrapped.device, dtype=torch.float32
                        ),
                        torch.tensor(
                            [view_target.tolist()], device=unwrapped.device, dtype=torch.float32
                        ),
                    )

                def set_manual_camera_view(
                    camera_path: str,
                    view_eye: np.ndarray,
                    view_target: np.ndarray,
                ) -> None:
                    from pxr import Gf, UsdGeom

                    camera_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
                        camera_path
                    )
                    xformable = UsdGeom.Xformable(camera_prim)
                    xformable.ClearXformOpOrder()
                    eye_gf = Gf.Vec3d(*map(float, view_eye))
                    target_gf = Gf.Vec3d(*map(float, view_target))
                    xformable.AddTransformOp().Set(
                        Gf.Matrix4d(1.0)
                        .SetLookAt(eye_gf, target_gf, Gf.Vec3d(0.0, 0.0, 1.0))
                        .GetInverse()
                    )

                followed_person = None
                if overview_sensor is not None and not args.go2_front_pinhole:
                    set_sensor_view(overview_sensor, eye, target)
                if joint_people_video:
                    official_count = len(roaming_assignments)
                    legacy_count = len(people_manager.agents) if people_manager is not None else 0
                    total_people_count = official_count or legacy_count
                    if total_people_count == 0:
                        raise RuntimeError("joint People capture requires a people manager")
                    if args.pedestrian_follow_index >= total_people_count:
                        raise IndexError(
                            "pedestrian follow index is outside configured People agents: "
                            f"{args.pedestrian_follow_index} >= {total_people_count}"
                        )
                    go2_eye, go2_target = capture.overview_camera_view(
                        position,
                        quaternion,
                        chase_distance=float(
                            roaming_payload.get("video", {})
                            .get("go2_follow_camera", {})
                            .get("chase_distance_m", 6.6)
                            if roaming_payload is not None
                            else 4.2
                        ),
                        lateral_offset=float(
                            roaming_payload.get("video", {})
                            .get("go2_follow_camera", {})
                            .get("lateral_offset_m", -4.0)
                            if roaming_payload is not None
                            else -3.0
                        ),
                        chase_height=float(
                            roaming_payload.get("video", {})
                            .get("go2_follow_camera", {})
                            .get("height_m", 6.8)
                            if roaming_payload is not None
                            else 3.2
                        ),
                        target_forward=float(
                            roaming_payload.get("video", {})
                            .get("go2_follow_camera", {})
                            .get("target_forward_m", 1.1)
                            if roaming_payload is not None
                            else 1.1
                        ),
                        target_lateral=0.0,
                        target_height=float(
                            roaming_payload.get("video", {})
                            .get("go2_follow_camera", {})
                            .get("target_height_m", 0.35)
                            if roaming_payload is not None
                            else 0.35
                        ),
                    )
                    go2_camera_payload = (
                        roaming_payload.get("video", {}).get(
                            "go2_follow_camera", {}
                        )
                        if roaming_payload is not None
                        else {}
                    )
                    if (
                        go2_camera_payload.get("fixed_eye_world_xyz") is not None
                        and go2_camera_payload.get("fixed_target_world_xyz") is not None
                    ):
                        go2_eye = np.asarray(
                            go2_camera_payload["fixed_eye_world_xyz"],
                            dtype=np.float64,
                        )
                        go2_target = np.asarray(
                            go2_camera_payload["fixed_target_world_xyz"],
                            dtype=np.float64,
                        )
                    if roaming_people_manager is not None:
                        index = int(
                            roaming_payload.get("video", {})
                            .get("person_follow_camera", {})
                            .get("resident_index", args.pedestrian_follow_index)
                        )
                        if not 0 <= index < len(latest_official_people_positions):
                            raise RuntimeError(
                                f"resident follow index {index} is outside the "
                                f"loaded People population of "
                                f"{len(latest_official_people_positions)}"
                            )
                        person_position = latest_official_people_positions[index]
                        person_delta = (
                            latest_official_people_positions[index]
                            - previous_official_people_positions[index]
                        )
                        if np.linalg.norm(person_delta[:2]) <= 1.0e-5:
                            person_delta = roaming_people_manager.targets[index] - person_position
                        person_heading = math.atan2(float(person_delta[1]), float(person_delta[0]))
                        followed_person = {
                            "name": roaming_assignments[index].name,
                            "position": person_position,
                            "heading": person_heading,
                        }
                    else:
                        followed_person = people_manager.agents[args.pedestrian_follow_index]
                    person_eye, person_target = pedestrian_follow_camera_view(
                        followed_person["position"],
                        float(followed_person["heading"]),
                        chase_distance=float(
                            roaming_payload.get("video", {})
                            .get("person_follow_camera", {})
                            .get("chase_distance_m", 10.0)
                            if roaming_payload is not None
                            else 4.0
                        ),
                        lateral_offset=float(
                            roaming_payload.get("video", {})
                            .get("person_follow_camera", {})
                            .get("lateral_offset_m", -1.5)
                            if roaming_payload is not None
                            else -3.0
                        ),
                        chase_height=float(
                            roaming_payload.get("video", {})
                            .get("person_follow_camera", {})
                            .get("height_m", 5.0)
                            if roaming_payload is not None
                            else 2.2
                        ),
                        target_forward=float(
                            roaming_payload.get("video", {})
                            .get("person_follow_camera", {})
                            .get("target_forward_m", 1.0)
                            if roaming_payload is not None
                            else 1.0
                        ),
                        target_height=float(
                            roaming_payload.get("video", {})
                            .get("person_follow_camera", {})
                            .get("target_height_m", 1.0)
                            if roaming_payload is not None
                            else 0.8
                        ),
                    )
                    person_camera_payload = (
                        roaming_payload.get("video", {}).get(
                            "person_follow_camera", {}
                        )
                        if roaming_payload is not None
                        else {}
                    )
                    if (
                        person_camera_payload.get("fixed_eye_world_xyz") is not None
                        and person_camera_payload.get("fixed_target_world_xyz") is not None
                    ):
                        person_eye = np.asarray(
                            person_camera_payload["fixed_eye_world_xyz"],
                            dtype=np.float64,
                        )
                        person_target = np.asarray(
                            person_camera_payload["fixed_target_world_xyz"],
                            dtype=np.float64,
                        )
                    if manual_two_panel_camera_paths:
                        set_manual_camera_view(
                            manual_two_panel_camera_paths["go2"], go2_eye, go2_target
                        )
                        set_manual_camera_view(
                            manual_two_panel_camera_paths["person"],
                            person_eye,
                            person_target,
                        )
                    else:
                        set_sensor_view(go2_follow_sensor, go2_eye, go2_target)
                        set_sensor_view(
                            person_follow_sensor, person_eye, person_target
                        )
                if road_sweep_payload is not None and not project_native_road_sweep:
                    (
                        road_sweep_eye,
                        road_sweep_target,
                        road_sweep_progress_m,
                        road_sweep_route_length_m,
                        road_sweep_reached_this_frame,
                    ) = polyline_camera_view(
                        timestamp,
                        road_sweep_eye_points,
                        float(road_sweep_payload["speed_mps"]),
                        float(road_sweep_payload["pitch_down_deg"]),
                        float(road_sweep_payload["lookahead_m"]),
                    )
                    set_manual_camera_view(
                        manual_two_panel_camera_paths["road_sweep"],
                        road_sweep_eye,
                        road_sweep_target,
                    )
                if args.joint_four_panel_video:
                    set_sensor_view(
                        scene_global_sensor,
                        np.asarray(args.scene_global_eye, dtype=np.float64),
                        np.asarray(args.scene_global_target, dtype=np.float64),
                    )

                carb.settings.get_settings().set(
                    "/rtx/post/tonemap/exposure", args.overview_exposure_ev
                )
                if overview_frame_count == 0:
                    print("RTX_DIAGNOSTIC phase=first_render_begin", flush=True)
                # Refresh Fabric/Hydra without advancing PhysX. Isaac Lab's
                # render() temporarily disables /app/player/playSimulations;
                # a raw SimulationApp.update() here would add an uncontrolled
                # physics step on every captured frame.
                if manual_two_panel_annotators:
                    # Manual post-reset render products are outside Isaac
                    # Lab's sensor registry.  Flush the live articulation into
                    # Fabric/Hydra first, then refresh the products explicitly.
                    # A positive delta is mandatory because NVIDIA's official
                    # DynamicObstacle computes velocity from the update delta.
                    unwrapped.sim.forward()
                    rep.orchestrator.step(
                        rt_subframes=1,
                        pause_timeline=False,
                        delta_time=dt,
                    )
                    if (
                        official_people_runtime is not None
                        and official_people_runtime.timeline is not None
                        and not official_people_runtime.timeline.is_playing()
                    ):
                        official_people_runtime.timeline.play()
                        official_people_runtime.timeline.commit()
                else:
                    unwrapped.sim.render()
                if overview_frame_count == 0:
                    print("RTX_DIAGNOSTIC phase=first_render_end", flush=True)
                if three_camera_writer is not None:
                    three_camera_writer.write(timestamp, step_index+1, position, quaternion,
                        overlap_labeler, overlap_input if overlap_labeler is not None else None)
                if overview_sensor is not None:
                    overview_sensor.update(0.0, force_recompute=True)
                if go2_follow_sensor is not None:
                    go2_follow_sensor.update(0.0, force_recompute=True)
                if person_follow_sensor is not None:
                    person_follow_sensor.update(0.0, force_recompute=True)
                if scene_global_sensor is not None:
                    scene_global_sensor.update(0.0, force_recompute=True)
                if overview_frame_count == 0:
                    print("RTX_DIAGNOSTIC phase=first_sensor_update_end", flush=True)

                def image_from_sensor(sensor: Any) -> Image.Image:
                    rgb = capture.normalise_rgb(sensor.data.output["rgb"][0])
                    rendered = Image.fromarray(rgb, mode="RGB")
                    # Isaac Lab Camera's off-screen RGB path does not honor the
                    # viewport tonemap exposure on Isaac Sim 4.5. Apply a
                    # deterministic visualization-only brightness compensation
                    # before annotations; sensor/physics state remains unchanged.
                    return ImageEnhance.Brightness(rendered).enhance(
                        args.overview_output_brightness
                    )

                def image_from_annotator(annotator: Any) -> Image.Image:
                    rgb = capture.normalise_rgb(annotator.get_data())
                    rendered = Image.fromarray(rgb, mode="RGB")
                    return ImageEnhance.Brightness(rendered).enhance(
                        args.overview_output_brightness
                    )

                image = (
                    image_from_sensor(overview_sensor)
                    if overview_sensor is not None
                    else None
                )
                if overview_frame_count == 0:
                    print("RTX_DIAGNOSTIC phase=first_rgb_readback_end", flush=True)
                go2_image = (
                    image_from_sensor(go2_follow_sensor)
                    if go2_follow_sensor is not None
                    else image_from_annotator(manual_two_panel_annotators["go2"])
                    if manual_two_panel_annotators
                    else None
                )
                person_image = (
                    image_from_sensor(person_follow_sensor)
                    if person_follow_sensor is not None
                    else image_from_annotator(
                        manual_two_panel_annotators["person"]
                    )
                    if manual_two_panel_annotators
                    else None
                )
                road_sweep_image = (
                    image
                    if project_native_road_sweep
                    else image_from_annotator(
                        manual_two_panel_annotators["road_sweep"]
                    )
                    if road_sweep_payload is not None
                    else None
                )
                scene_global_image = (
                    image_from_sensor(scene_global_sensor)
                    if scene_global_sensor is not None
                    else None
                )
                draw = ImageDraw.Draw(image, "RGBA") if image is not None else None
                traffic_counts = (
                    {state: sum(a["status"] == state for a in traffic_manager.agents) for state in ("moving", "arrived", "scheduled", "failed")}
                    if traffic_manager is not None
                    else {}
                )
                driving_count = (
                    sum(
                        agent["status"] == "moving" and float(agent.get("speed", 0.0)) > 0.25
                        for agent in traffic_manager.agents
                    )
                    if traffic_manager is not None
                    else 0
                )
                nearby_ids = traffic_manager.third_person_pass_current_ids if traffic_manager is not None else []
                walking_people = (
                    sum(float(agent.get("speed", 0.0)) > 0.02 for agent in people_manager.agents)
                    if people_manager is not None
                    else int(
                        np.count_nonzero(
                            np.linalg.norm(
                                latest_official_people_positions[:, :2]
                                - previous_official_people_positions[:, :2],
                                axis=1,
                            )
                            > 1.0e-4
                        )
                    )
                )
                people_count_for_overlay = (
                    len(people_manager.agents)
                    if people_manager is not None
                    else len(latest_official_people_positions)
                )
                if draw is not None:
                    draw.rounded_rectangle(
                        (10, 10, 330, 40), radius=6, fill=(8, 12, 18, 165)
                    )
                    draw.text(
                        (18, 18),
                        f"t={timestamp:05.1f}s  active={traffic_counts.get('moving', 0)} "
                        f"driving={driving_count} "
                        f"people={walking_people}/{people_count_for_overlay} "
                        f"near={nearby_ids or '-'}",
                        fill=(180, 230, 255),
                    )
                if draw is not None and args.overview_motion_start_eye is not None:
                    draw.rounded_rectangle(
                        (10, 46, 410, 72), radius=5, fill=(8, 12, 18, 165)
                    )
                    draw.text(
                        (18, 53),
                        f"MOVING CAMERA  Go2 progress={arc_progress:05.2f}m "
                        f"speed={max(0.0, float(velocity_body[0])):0.2f}m/s",
                        fill=(240, 175, 255),
                    )
                if joint_people_video:
                    if go2_image is None or person_image is None or followed_person is None:
                        raise RuntimeError("joint People camera images were not produced")
                    if draw is not None:
                        draw.rounded_rectangle(
                            (10, 46, 92, 72), radius=5, fill=(8, 12, 18, 165)
                        )
                        draw.text((18, 53), "LOCAL GLOBAL", fill=(245, 245, 245))

                    go2_draw = ImageDraw.Draw(go2_image, "RGBA")
                    go2_draw.rounded_rectangle(
                        (10, 10, 112, 40), radius=6, fill=(8, 12, 18, 165)
                    )
                    go2_fixed = bool(
                        roaming_payload is not None
                        and roaming_payload.get("video", {})
                        .get("go2_follow_camera", {})
                        .get("fixed_eye_world_xyz") is not None
                    )
                    go2_draw.text(
                        (18, 18),
                        "GO2 FIXED" if go2_fixed else "GO2 FOLLOW",
                        fill=(190, 255, 205),
                    )

                    person_draw = ImageDraw.Draw(person_image, "RGBA")
                    person_draw.rounded_rectangle(
                        (10, 10, 178, 40), radius=6, fill=(8, 12, 18, 165)
                    )
                    person_draw.text(
                        (18, 18),
                        (
                            "PEOPLE FIXED"
                            if roaming_payload is not None
                            and roaming_payload.get("video", {})
                            .get("person_follow_camera", {})
                            .get("fixed_eye_world_xyz") is not None
                            else f"PERSON FOLLOW: {followed_person['name']}"
                        ),
                        fill=(255, 225, 175),
                    )

                    if args.joint_four_panel_video:
                        if scene_global_image is None or image is None:
                            raise RuntimeError("whole-scene global image was not produced")
                        scene_draw = ImageDraw.Draw(scene_global_image, "RGBA")
                        scene_draw.rounded_rectangle(
                            (10, 10, 142, 40), radius=6, fill=(8, 12, 18, 165)
                        )
                        scene_draw.text(
                            (18, 18), "SCENE GLOBAL", fill=(245, 245, 245)
                        )
                        output_image = Image.new(
                            "RGB",
                            (args.overview_width * 2, args.overview_height * 2),
                            (8, 12, 18),
                        )
                        output_image.paste(scene_global_image, (0, 0))
                        output_image.paste(image, (args.overview_width, 0))
                        output_image.paste(go2_image, (0, args.overview_height))
                        output_image.paste(
                            person_image,
                            (args.overview_width, args.overview_height),
                        )
                    elif args.roaming_ghost_two_panel_video:
                        output_image = Image.new(
                            "RGB",
                            (args.overview_width * 2, args.overview_height),
                            (8, 12, 18),
                        )
                        output_image.paste(go2_image, (0, 0))
                        output_image.paste(person_image, (args.overview_width, 0))
                    else:
                        if image is None:
                            raise RuntimeError("global overview image was not produced")
                        output_image = Image.new(
                            "RGB",
                            (args.overview_width * 3, args.overview_height),
                            (8, 12, 18),
                        )
                        output_image.paste(image, (0, 0))
                        output_image.paste(go2_image, (args.overview_width, 0))
                        output_image.paste(person_image, (args.overview_width * 2, 0))
                else:
                    output_image = image
                frame = np.asarray(output_image)
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if overlap_recorder is not None:
                    frame_record = overlap_record
                    if args.go2_front_pinhole:
                        camera_world = overview_sensor.data.pos_w[0].detach().cpu().numpy().astype(float)
                        # Camera's XFormPrim pose readback can lag the live Fabric
                        # root by one render interval. Use the rigid mount model
                        # at this control timestamp, not that stale USD view.
                        frame_record = dict(overlap_record,
                            camera_pose_source='rigid_mount_full_root_pose_at_capture_not_pixel_pose_proof',
                            native_camera_pose_diagnostic=camera_world.tolist(),
                            native_pose_discrepancy_m=float(np.linalg.norm(camera_world-overlap_camera)))
                    overlap_recorder.frame(frame_record, overview_frame_count, timestamp_s=timestamp,
                        camera_id='front_pinhole' if args.go2_front_pinhole else 'overview_composite_camera_not_evaluated')
                if overview_frame_count == 0 or overview_frame_count % max(1, int(actual_overview_fps * 10)) == 0:
                    key_path = captures_dir / f"overview_frame_{overview_frame_count:05d}.png"
                    output_image.save(key_path)
                    overview_keyframes.append(key_path)
                    if project_native_road_sweep:
                        road_sweep_keyframes.append(key_path)
                overview_frame_count += 1
                if project_native_road_sweep:
                    road_sweep_frame_count = overview_frame_count
                    if road_sweep_reached_this_frame:
                        road_sweep_endpoint_reached = True
                        if road_sweep_completion_time_s is None:
                            road_sweep_completion_time_s = timestamp
                if (
                    road_sweep_video_writer is not None
                    and road_sweep_image is not None
                ):
                    road_sweep_frame = np.asarray(road_sweep_image)
                    road_sweep_video_writer.write(
                        cv2.cvtColor(road_sweep_frame, cv2.COLOR_RGB2BGR)
                    )
                    save_road_sweep_keyframe = bool(
                        road_sweep_frame_count == 0
                        or road_sweep_frame_count
                        % max(1, int(actual_overview_fps * 10))
                        == 0
                        or road_sweep_reached_this_frame
                    )
                    if save_road_sweep_keyframe:
                        road_sweep_key_path = (
                            captures_dir
                            / f"road_sweep_frame_{road_sweep_frame_count:05d}.png"
                        )
                        road_sweep_image.save(road_sweep_key_path)
                        road_sweep_keyframes.append(road_sweep_key_path)
                    road_sweep_frame_count += 1
                    if road_sweep_reached_this_frame:
                        road_sweep_endpoint_reached = True
                        road_sweep_completion_time_s = timestamp
                        road_sweep_video_writer.release()
                        road_sweep_video_writer = None
                overview_preview_captured = bool(args.overview_preview_time_s is not None)
            if step_index % 5 == 0 or step_index == total_steps - 1:
                append_jsonl(
                    trajectory_handle,
                    {
                        "step_index": step_index + 1,
                        "timestamp_s": timestamp,
                        "mode": args.mode,
                        "command": command,
                        "base_position_world": position.tolist(),
                        "base_quaternion_wxyz": quaternion.tolist(),
                        "motion_segment_id": recovery.segment if recovery is not None else 0,
                        "base_yaw_rad_world": yaw,
                        "base_linear_velocity_body": velocity_body.tolist(),
                        "base_angular_velocity_body": angular_body.tolist(),
                        "route_arc_progress_m": arc_progress,
                        "cross_track_error_m": cross_track,
                        "contact_classification": contact_state,
                        "stuck_detection": stuck_state,
                        "sustained_collision_detection": collision_state,
                        "scene10_multivehicle_traffic": traffic_state,
                        "pedestrian_positions_xyz": (
                            latest_official_people_positions.tolist()
                            if len(latest_official_people_positions)
                            else []
                        ),
                        "pedestrian_ids": [item.name for item in roaming_assignments],
                        "micromobility_states": {
                            agent_id: {
                                "position_xy": state.position_xy.tolist(),
                                "yaw_rad": float(state.yaw_rad),
                                "speed_mps": float(state.speed_mps),
                            }
                            for agent_id, state in micromobility_states.items()
                        },
                    },
                )
            if overview_preview_captured:
                stop_reason = "overview_framing_preview_captured"
                break
            if project_native_road_sweep and road_sweep_endpoint_reached:
                stop_reason = "road_sweep_endpoint_reached"
                break
            tolerate_dynamic_contact=False
            if recovery is not None:
                circles=dynamic_circles(traffic_manager.agents,people_manager.positions(),micromobility_states,micro_catalog,micromobility_specs)
                recovery.observe(timestamp,position,nonfoot,circles)
                recent_dynamic=timestamp-recovery.last_overlap<=5.
                fallen=bool(float(robot.data.projected_gravity_b[0,2])>-.45 or position[2]<float(route_payload['ground_z'])+.17)
                if (recent_dynamic and (fallen or stuck_state['stuck'])) or recovery.waiting_since is not None:
                    if len(recovery.events)>=recovery.settings.get('maximum_recoveries',8):
                        stop_reason='forward_recovery_budget_exhausted';route_stuck=True;break
                    proposal=recovery.proposal(arc_progress,circles)
                    if proposal is None:
                        if recovery.waiting_since is None:recovery.waiting_since=timestamp
                        if timestamp-recovery.waiting_since>5.:
                            stop_reason='no_safe_forward_recovery_point';route_stuck=True;break
                        continue
                    event=recovery.apply(robot,external,controller,unwrapped.sim,unwrapped.action_manager,proposal,timestamp,position,
                        'fall after dynamic proximity/contact' if fallen else 'stall after dynamic proximity/contact')
                    write_json(metadata_dir/'forward_recovery_events.json',recovery.events)
                    print('FORWARD_RECOVERY '+json.dumps(event),flush=True)
                    observations=unwrapped.observation_manager.compute()
                    stuck_monitor.reset();collision_monitor.reset()
                    continue
                tolerate_dynamic_contact=recent_dynamic and not fallen and not stuck_state['stuck']
            if collision_state["sustained"] and not tolerate_dynamic_contact:
                sustained_collision = True
                stop_reason = "sustained_nonfoot_collision"
                break
            if stuck_state["stuck"]:
                route_stuck = True
                stop_reason = "confirmed_stuck"
                break
            if args.mode == "route" and cross_track > args.maximum_cross_track_error:
                route_diverged = True
                stop_reason = "excessive_cross_track_error"
                break
            if controller is not None and controller.route_complete:
                goal_reached = True
                if goal_reached_at_s is None:
                    goal_reached_at_s = timestamp
                if args.stop_at_route_goal:
                    stop_reason = "route_goal_reached"
                    break
                if roaming_payload is not None:
                    continue
                continuous_ready = bool(
                    traffic_manager is not None
                    and getattr(traffic_manager, "continuous_looping", False)
                    and min(agent["spawn_count"] for agent in traffic_manager.agents) >= 1
                    and (
                        not args.overview_video
                        or traffic_manager.third_person_pass_max_s
                        >= args.minimum_visible_vehicle_pass_duration
                    )
                )
                if (
                    traffic_manager is None
                    or traffic_manager.all_terminal
                    or continuous_ready
                ):
                    stop_reason = (
                        "route_goal_with_continuous_traffic"
                        if continuous_ready
                        else (
                            "route_goal_and_all_vehicles_terminal"
                            if traffic_manager is not None
                            else "route_goal_reached"
                        )
                    )
                    break
            if exit_controller is not None and exit_controller.route_complete:
                goal_reached = True
                stop_reason = "three_point_goal_reached"
                break
        run_wall_s = time.perf_counter() - run_started
        if roaming_people_manager is not None and actual_positions:
            final_sample_time_s = len(actual_positions) * dt
            if (
                not official_people_runtime_samples
                or abs(
                    float(official_people_runtime_samples[-1]["simulation_time_s"])
                    - final_sample_time_s
                )
                > 0.5 * dt
            ):
                official_people_runtime_samples.append(
                    {
                        "simulation_time_s": final_sample_time_s,
                        "positions_xy": latest_official_people_positions[:, :2].tolist(),
                        "maximum_displacement_from_start_m": (
                            maximum_official_people_displacement_m
                        ),
                        "official_timeline_time_s": float(
                            official_people_runtime.timeline.get_current_time()
                        ),
                    }
                )
        if open_invalid_labels and open_invalid_start_s is not None:
            invalid_intervals.append(
                {
                    "start_s": open_invalid_start_s,
                    "end_s": len(actual_positions) * dt,
                    "labels": list(open_invalid_labels),
                    "ordinary_navigation_training_valid": False,
                }
            )
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        if road_sweep_video_writer is not None:
            road_sweep_video_writer.release()
            road_sweep_video_writer = None
        trajectory_handle.close()
        trajectory_handle = None
        if overlap_recorder is not None:
            overlap_recorder.close(completed=True)
        actual = np.asarray(actual_positions, dtype=np.float64)
        command_array = np.asarray(commands, dtype=np.float64)
        body_linear_array = np.asarray(body_linear_samples, dtype=np.float64)
        body_angular_array = np.asarray(body_angular_samples, dtype=np.float64)
        segments=np.asarray(actual_segments)
        walked=np.linalg.norm(np.diff(actual[:,:2],axis=0),axis=1) if len(actual)>1 else np.array([])
        walked=walked*(np.diff(segments)==0)
        path_length=float(walked.sum())
        tail_count = min(len(actual), int(round(10.0 / dt)))
        tail_path=float(walked[-(tail_count-1):].sum()) if tail_count>1 else 0.
        if recovery is not None:
            write_json(metadata_dir/'forward_recovery_events.json',recovery.events)
            contact_query_end=dynamic_contact_runtime.audit_physx(omni.physx.get_physx_scene_query_interface())
            write_json(metadata_dir/'dynamic_contact_physx_query_end.json',contact_query_end)
            write_json(metadata_dir/'forward_recovery_summary.json',dict(enabled=True,recovery_count=len(recovery.events),
                reset_segments=recovery.segment,skipped_arc_m=sum(e['skipped_arc_m'] for e in recovery.events),
                walked_distance_excludes_resets=True,
                contact_correlated_resets=sum(bool(e['physical_contact_recent']) for e in recovery.events),
                post_reset_physical_walked_m={str(s):float(walked[segments[1:]==s].sum()) for s in range(1,recovery.segment+1)},
                rtx_recovery_verified=False))
        constant_tracking = None
        traffic_summary = traffic_manager.summary() if traffic_manager is not None else None
        if traffic_summary is not None and mixed_roaming_payload is not None:
            traffic_summary["go2_interaction"] = (
                "ignored by vehicle motion and avoidance; distance is diagnostic only"
                if dynamic_agents_ignore_go2
                else "treated as an external yield obstacle"
            )
        people_summary = (
            people_manager.summary()
            if people_manager is not None
            else roaming_people_manager.metrics()
            if roaming_people_manager is not None
            else None
        )
        if people_summary is not None and roaming_people_manager is not None:
            people_summary["maximum_displacement_from_start_m"] = (
                maximum_official_people_displacement_m
            )
            people_summary["runtime_sample_count"] = len(
                official_people_runtime_samples
            )
            write_json(
                metadata_dir / "official_people_runtime_samples.json",
                official_people_runtime_samples,
            )
        elif people_summary is not None and mixed_roaming_payload is not None:
            people_summary["maximum_displacement_from_start_m"] = (
                maximum_official_people_displacement_m
            )
        micromobility_summary = (
            micromobility_manager.metrics()
            if micromobility_manager is not None
            else None
        )
        if micromobility_summary is not None and micromobility_usd is not None:
            micromobility_summary["go2_interaction"] = (
                "ignored by micromobility motion and avoidance"
                if dynamic_agents_ignore_go2
                else "treated as an external avoidance obstacle"
            )
            micromobility_summary["runtime_visual_bounds"] = (
                micromobility_usd.visual_bounds()
            )
            micromobility_summary["runtime_support_grounding"] = (
                micromobility_usd.support_query_summary()
            )
        mixed_walkable_summary = (
            mixed_walkable_audit.summary()
            if mixed_walkable_audit is not None
            else None
        )
        official_dynamic_obstacle_runtime = None
        if micromobility_usd is not None and roaming_people_manager is not None:
            from omni.anim.people.scripts.global_character_position_manager import (
                GlobalCharacterPositionManager,
            )

            official_position_manager = GlobalCharacterPositionManager.get_instance()
            managed_paths = {
                str(value)
                for value in official_position_manager.get_all_managed_characters()
            }
            expected_paths = {
                f"{micromobility_usd.parent_path}/{spec.agent_id}"
                for spec in micromobility_specs
            }
            official_dynamic_obstacle_runtime = {
                "script_count": len(micromobility_usd.dynamic_obstacle_scripts),
                "expected_paths": sorted(expected_paths),
                "managed_paths": sorted(managed_paths),
                "all_expected_paths_managed": expected_paths.issubset(managed_paths),
                "managed_radii_m": {
                    path: float(official_position_manager.get_character_radius(path))
                    for path in sorted(expected_paths & managed_paths)
                },
            }
        crossed_static_obstacle_ids = {
            label.removeprefix("static:")
            for interval in invalid_intervals
            for label in interval["labels"]
            if label.startswith("static:")
        }
        ghost_static_categories_crossed = sorted(
            {
                obstacle.category
                for obstacle in (
                    ghost_experiment.obstacles if ghost_experiment is not None else ()
                )
                if obstacle.obstacle_id in crossed_static_obstacle_ids
            }
        )
        video_validation = None
        road_sweep_video_validation = None
        if capture_driver_sensor is not None:
            import cv2

            cap = cv2.VideoCapture(str(overview_video_path))
            video_validation = {
                "opened": bool(cap.isOpened()),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "path": str(overview_video_path),
                "codec": args.overview_video_codec,
                "h264_crf": (
                    args.overview_h264_crf
                    if args.overview_video_codec == "h264"
                    else None
                ),
                "h264_preset": (
                    args.overview_h264_preset
                    if args.overview_video_codec == "h264"
                    else None
                ),
            }
            cap.release()
            if road_sweep_video_path is not None:
                road_sweep_cap = cv2.VideoCapture(str(road_sweep_video_path))
                road_sweep_video_validation = {
                    "opened": bool(road_sweep_cap.isOpened()),
                    "frame_count": int(
                        road_sweep_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    ),
                    "width": int(road_sweep_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(
                        road_sweep_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    ),
                    "fps": float(road_sweep_cap.get(cv2.CAP_PROP_FPS)),
                    "path": str(road_sweep_video_path),
                    "codec": args.overview_video_codec,
                    "frame_count_written": road_sweep_frame_count,
                    "endpoint_reached": road_sweep_endpoint_reached,
                    "completion_time_s": road_sweep_completion_time_s,
                    "route_progress_m": road_sweep_progress_m,
                    "route_length_m": road_sweep_route_length_m,
                    "expected_duration_s": (
                        road_sweep_route_length_m
                        / float(road_sweep_payload["speed_mps"])
                        if road_sweep_route_length_m is not None
                        else None
                    ),
                    "camera_config": road_sweep_payload,
                    "keyframes": [str(path) for path in road_sweep_keyframes],
                }
                road_sweep_cap.release()
        if args.overview_preview_time_s is not None:
            passed = bool(
                overview_preview_captured
                and overview_frame_count == 1
                and len(overview_keyframes) == 1
                and video_validation is not None
                and video_validation["opened"]
                and video_validation["frame_count"] == 1
            )
        elif args.mode in ("route", "three_point"):
            if ghost_mode_enabled and args.ghost_ab_normal:
                # The A/B baseline is successful only when the same frozen
                # policy is physically stopped by an explicit experiment
                # obstacle.  Treating the expected collision/stuck signal as a
                # generic route failure made the normal leg impossible to pass
                # and weakened the contrast with the ghost leg.
                passed = bool(
                    stop_reason in ("sustained_nonfoot_collision", "confirmed_stuck")
                    and path_length >= 0.50
                    and foot_support_steps / max(1, len(actual)) >= 0.90
                )
            elif roaming_payload is not None:
                # Joint resident-agent showcases may intentionally use a route
                # whose endpoint lies beyond the requested capture duration.
                # This duration-based acceptance is independent of whether the
                # optional Go2 ghost collision experiment is enabled.
                passed = bool(
                    stop_reason
                    in (
                        "duration_complete",
                        "route_goal_reached",
                        "road_sweep_endpoint_reached",
                    )
                    and path_length >= 0.25
                    and not route_stuck
                    and not route_diverged
                    and not sustained_collision
                    and foot_support_steps / max(1, len(actual)) >= 0.90
                )
            else:
                passed = (
                    (goal_reached or args.pedestrian_lifecycle_validation)
                    and not route_stuck
                    and not sustained_collision
                )
            if traffic_summary is not None:
                visual_rows = traffic_summary.get("visual_geometry_alignment", [])
                # Vehicles waiting for their next lifecycle spawn are deliberately
                # parked off-map and have no composed render mesh.  They are not
                # valid subjects for the on-route visual-alignment acceptance test.
                active_visual_rows = [
                    row
                    for row in visual_rows
                    if row.get("expected_location") == "route"
                ]
                fast_proxy_alignment_passed = bool(
                    active_visual_rows
                    and all(
                        row.get("bbox_valid")
                        and row.get("runtime_proxy_front_heading_deg") is not None
                        and row.get("bbox_to_expected_visual_center_xy_error_m") is not None
                        and row["bbox_to_expected_visual_center_xy_error_m"] <= 0.1
                        for row in active_visual_rows
                    )
                    and traffic_summary["maximum_heading_motion_error_deg"] <= 0.5
                )
                rendered_body_alignment_passed = (
                    (
                        traffic_summary["maximum_rendered_body_axis_error_deg"] is not None
                        and traffic_summary["maximum_rendered_body_axis_error_deg"] <= 3.0
                    )
                    or fast_proxy_alignment_passed
                    if args.overview_video
                    else True
                )
                if traffic_summary.get("continuous_looping"):
                    traffic_lifecycle_passed = bool(
                        min(traffic_summary["spawn_counts"]) >= 1
                        and traffic_summary["maximum_heading_motion_error_deg"] <= 0.5
                        and rendered_body_alignment_passed
                        and (
                            # A low-cost smoke admission only needs evidence that
                            # the resident traffic is live.  Requiring every car
                            # to move simultaneously incorrectly rejects valid
                            # bottleneck/junction scheduling where some cars must
                            # yield while others proceed.
                            traffic_summary["maximum_simultaneously_moving_vehicle_count"]
                            >= 1
                            if args.roaming_validation_level == "smoke"
                            else traffic_summary["total_completed_cycles"] >= 1
                        )
                    )
                else:
                    traffic_lifecycle_passed = traffic_summary["all_vehicles_arrived"]
                passed = bool(
                    passed
                    and traffic_lifecycle_passed
                    and traffic_summary["dynamic_obb_overlap_events"] == 0
                    and traffic_summary["static_obb_overlap_events"] == 0
                    and (
                        dynamic_agents_ignore_go2
                        or ghost_mode_enabled
                        or (
                            traffic_summary["go2_vehicle_overlap_events"] == 0
                            and traffic_summary["minimum_go2_vehicle_clearance_m"] is not None
                            and traffic_summary["minimum_go2_vehicle_clearance_m"]
                            >= args.minimum_go2_vehicle_clearance
                        )
                    )
                )
                if args.overview_video and args.overview_fixed_eye is None:
                    passed = bool(
                        passed
                        and traffic_summary["third_person_dynamic_pass_max_duration_s"]
                        >= args.minimum_visible_vehicle_pass_duration
                    )
            if people_summary is not None:
                if roaming_people_manager is not None or mixed_roaming_payload is not None:
                    completed_trip_values = list(
                        people_summary["completed_trips"].values()
                        if isinstance(people_summary["completed_trips"], dict)
                        else people_summary["completed_trips"]
                    )
                    expected_people_count = int(
                        roaming_payload[
                            "people" if mixed_roaming_payload is not None else "official_people"
                        ]["count"]
                    )
                    people_lifecycle_passed = bool(
                        people_summary["resident_count"]
                        == expected_people_count
                        and (
                            args.roaming_validation_level == "smoke"
                            or sum(completed_trip_values) >= 1
                        )
                    )
                    passed = bool(passed and people_lifecycle_passed)
                elif people_summary.get("continuous_spawn_despawn"):
                    route_schedulers = people_summary["route_schedulers"].values()
                    people_lifecycle_passed = bool(
                        people_summary["maximum_active_pedestrian_count"] >= 8
                        and all(row["spawn_count"] >= 3 for row in route_schedulers)
                        and all(row["completion_count"] >= 1 for row in route_schedulers)
                        and people_summary["maximum_heading_motion_error_deg"] <= 0.5
                        and people_summary["maximum_turn_angle_deg"]
                        <= people_summary["maximum_turn_rate_deg_s"] * 0.02 + 1.0e-6
                        and (
                            people_summary["minimum_pedestrian_clearance_m"] is None
                            or people_summary["minimum_pedestrian_clearance_m"] >= -0.05
                        )
                    )
                    passed = bool(passed and people_lifecycle_passed)
                else:
                    passed = bool(
                        passed
                        and people_summary["maximum_simultaneously_moving_pedestrian_count"]
                        == people_summary["pedestrian_count"]
                        and min(people_summary["path_lengths_m"].values()) >= 0.5
                    )
            if roaming_people_manager is not None or mixed_roaming_payload is not None:
                required_continuous_targets = args.roaming_validation_level in (
                    "technical",
                    "formal",
                )
                mixed_gate = bool(
                    mixed_walkable_summary is not None
                    and mixed_walkable_summary["people_component_violations"] == 0
                    and mixed_walkable_summary[
                        "people_approved_union_violations"
                    ]
                    == 0
                    and mixed_walkable_summary[
                        "micromobility_component_violations"
                    ]
                    == 0
                    and mixed_walkable_summary[
                        "micromobility_obb_overlap_events"
                    ]
                    == 0
                    and mixed_walkable_summary[
                        "people_people_severe_overlap_events"
                    ]
                    == 0
                    and (
                        mixed_walkable_summary[
                            "minimum_people_micromobility_clearance_m"
                        ]
                        is None
                        or mixed_walkable_summary[
                            "minimum_people_micromobility_clearance_m"
                        ]
                        >= -0.03
                    )
                    and mixed_walkable_summary["maximum_people_displacement_m"]
                    >= 0.30
                    and (
                        not micromobility_specs
                        or mixed_walkable_summary[
                            "maximum_micromobility_displacement_m"
                        ]
                        >= 0.50
                    )
                )
                official_obstacle_gate = bool(
                    mixed_roaming_payload is not None
                    or not micromobility_specs
                    or (
                        official_dynamic_obstacle_runtime is not None
                        and official_dynamic_obstacle_runtime[
                            "all_expected_paths_managed"
                        ]
                        and official_dynamic_obstacle_runtime["script_count"]
                        == len(micromobility_specs)
                    )
                )
                completed_trip_values = list(
                    people_summary["completed_trips"].values()
                    if isinstance(people_summary["completed_trips"], dict)
                    else people_summary["completed_trips"]
                )
                target_refresh_gate = bool(
                    not required_continuous_targets
                    or (
                        (
                            min(completed_trip_values) >= 1
                            if args.roaming_validation_level == "formal"
                            else sum(completed_trip_values) >= 1
                        )
                        and (
                            not micromobility_specs
                            or (
                                micromobility_summary is not None
                                and (
                                    min(
                                        micromobility_summary[
                                            "completed_tasks"
                                        ].values()
                                    )
                                    >= 1
                                    if args.roaming_validation_level == "formal"
                                    else sum(
                                        micromobility_summary[
                                            "completed_tasks"
                                        ].values()
                                    )
                                    >= 1
                                )
                            )
                        )
                    )
                )
                if args.roaming_validation_level == "formal" and micromobility_specs:
                    mixed_gate = bool(
                        mixed_gate
                        and mixed_walkable_summary["close_interaction_samples"] > 0
                    )
                ghost_showcase_gate = bool(
                    not ghost_mode_enabled
                    or args.roaming_validation_level != "formal"
                    or len(ghost_static_categories_crossed) >= 2
                )
                passed = bool(
                    passed
                    and mixed_gate
                    and official_obstacle_gate
                    and target_refresh_gate
                    and ghost_showcase_gate
                    and (
                        mixed_roaming_payload is not None
                        or (
                            official_people_collision_audit is not None
                            and official_people_collision_audit["passed"]
                        )
                    )
                )
            if video_validation is not None:
                passed = bool(
                    passed
                    and video_validation["opened"]
                    and video_validation["frame_count"] == overview_frame_count
                )
            if road_sweep_payload is not None:
                passed = bool(
                    passed
                    and road_sweep_video_validation is not None
                    and road_sweep_video_validation["opened"]
                    and road_sweep_video_validation["frame_count"]
                    == road_sweep_frame_count
                    and road_sweep_frame_count > 0
                    and (
                        args.roaming_validation_level != "formal"
                        or road_sweep_endpoint_reached
                    )
                )
        elif args.mode == "constant":
            tail_linear_x = float(np.median(body_linear_array[-tail_count:, 0]))
            tail_angular_z = float(np.median(body_angular_array[-tail_count:, 2]))
            if abs(args.constant_vx) < 1.0e-6:
                linear_ok = abs(tail_linear_x) <= 0.05
            else:
                linear_ok = tail_linear_x * math.copysign(1.0, args.constant_vx) >= max(
                    0.025, 0.20 * abs(args.constant_vx)
                )
            if abs(args.constant_wz) < 1.0e-6:
                angular_ok = abs(tail_angular_z) <= 0.08
            else:
                angular_ok = tail_angular_z * math.copysign(1.0, args.constant_wz) >= max(
                    0.015, 0.20 * abs(args.constant_wz)
                )
            constant_tracking = {
                "tail_median_body_linear_x_mps": tail_linear_x,
                "tail_median_body_angular_z_radps": tail_angular_z,
                "linear_command_tracked": linear_ok,
                "angular_command_tracked": angular_ok,
            }
            passed = (
                linear_ok
                and angular_ok
                and (tail_path >= 0.25 or abs(args.constant_vx) < 1.0e-6)
                and not route_stuck
                and not sustained_collision
            )
        else:
            passed = tail_path >= 0.25 and not route_stuck and not sustained_collision
        if recovery is not None:
            passed = passed and contact_query_end['passed']
        if args.stop_at_route_goal:
            passed = passed and goal_reached
        status = "passed" if passed else "failed"
        plot_path = visualizations_dir / f"scene10_{args.mode}_trajectory.png"
        make_trajectory_plot(plot_path, args.mode, route_points, actual, status, segments)
        Image.open(plot_path).convert("RGB").save(captures_dir / "trajectory_diagnostic_rgb.png")
        mixed_actual_trajectory_map = None
        if mixed_roaming_payload is not None:
            mixed_actual_trajectory_map = build_mixed_actual_trajectory_map(
                visualizations_dir / "mixed_actual_trajectories.png",
                trajectory_path,
                mixed_roaming_config_path,
                args.traffic_scene_config,
            )
        summary = {
            "status": status,
            "mode": args.mode,
            "stop_reason": stop_reason,
            "goal_reached": goal_reached,
            "goal_reached_at_s": goal_reached_at_s,
            "confirmed_stuck": route_stuck,
            "route_diverged": route_diverged,
            "sustained_nonfoot_collision": sustained_collision,
            "requested_duration_s": duration_s,
            "stop_at_route_goal": args.stop_at_route_goal,
            "simulation_duration_s": len(actual) * dt,
            "simulation_wall_s": run_wall_s,
            "end_to_end_wall_s": time.perf_counter() - started,
            "path_length_m": path_length,
            "tail_10s_path_length_m": tail_path,
            "final_route_arc_progress_m": closest_route_metrics(actual[-1, :2], route_points, route_arc)[0],
            "maximum_cross_track_error_m": maximum_cross_track,
            "maximum_tracking_curvature_rad_per_m": maximum_tracking_curvature,
            "nonfoot_contact_step_count": nonfoot_steps,
            "head_contact_step_count": head_contact_steps,
            "foot_support_fraction": foot_support_steps / max(1, len(actual)),
            "go2_root_pose_direct_write": {
                "count": 0,
                "evidence": (
                    "robot.data.root_pos_w is only read (telemetry, controller, "
                    "overlap labels); set_world_poses_from_view is used solely for "
                    "camera sensors; root motion comes from frozen policy + env.step"
                ),
            },
            "fabric_render_bridge": joint_render_bridge_settings,
            "command_statistics": {
                "vx_min": float(np.min(command_array[:, 0])),
                "vx_median": float(np.median(command_array[:, 0])),
                "vx_max": float(np.max(command_array[:, 0])),
                "wz_min": float(np.min(command_array[:, 2])),
                "wz_median": float(np.median(command_array[:, 2])),
                "wz_max": float(np.max(command_array[:, 2])),
            },
            "constant_command_tracking": constant_tracking,
            "controller_configuration": {
                "lookahead_distance_m": args.lookahead_distance,
                "max_forward_speed_mps": args.max_forward_speed,
                "minimum_tracking_speed_mps": args.minimum_tracking_speed,
                "max_tracking_yaw_rate_radps": args.max_tracking_yaw_rate,
                "curvature_speed_gain": args.curvature_speed_gain,
                "yaw_command_gain": args.yaw_command_gain,
                "maximum_cross_track_error_m": args.maximum_cross_track_error,
            },
            "policy_kind": args.policy_kind,
            "go2_highlight_material": go2_highlight_metadata,
            "go2_runtime_physics_support": go2_support_metadata,
            "actor": actor_metadata,
            "replay": replay_meta,
            "maneuver_plan": str(args.maneuver_plan.resolve()) if args.maneuver_plan else None,
            "scene10_multivehicle_traffic": traffic_summary,
            "animated_pedestrians": people_summary,
            "official_resident_roaming": roaming_people_manager is not None,
            "project_resident_roaming": mixed_roaming_payload is not None,
            "dynamic_agents_ignore_go2": dynamic_agents_ignore_go2,
            "mixed_actual_trajectory_map": mixed_actual_trajectory_map,
            "official_people_collision_audit": official_people_collision_audit,
            "roaming_validation_level": args.roaming_validation_level,
            "approved_walking_navmesh": approved_navmesh_metadata,
            "micromobility": micromobility_summary,
            "mixed_walkable_agent_audit": mixed_walkable_summary,
            "mixed_reciprocal_avoidance_audit": (
                mixed_avoidance_audit if mixed_roaming_payload is not None else None
            ),
            "official_micromobility_dynamic_obstacles": (
                official_dynamic_obstacle_runtime
            ),
            "ghost_obstacles": ghost_obstacle_metadata,
            "ghost_training_invalid_intervals": invalid_intervals,
            "ghost_training_invalid_interval_count": len(invalid_intervals),
            "ghost_static_categories_crossed": ghost_static_categories_crossed,
            "ghost_ab_expectation": (
                {
                    "leg": "normal_collision",
                    "expected_terminal_reason": [
                        "sustained_nonfoot_collision",
                        "confirmed_stuck",
                    ],
                    "observed_terminal_reason": stop_reason,
                    "blocked_as_expected": bool(
                        stop_reason
                        in ("sustained_nonfoot_collision", "confirmed_stuck")
                    ),
                }
                if ghost_mode_enabled and args.ghost_ab_normal
                else {
                    "leg": "ghost_passthrough",
                    "expected_terminal_reason": [
                        "duration_complete",
                        "route_goal_reached",
                    ],
                    "observed_terminal_reason": stop_reason,
                    "blocked_as_expected": False,
                }
                if ghost_mode_enabled
                else None
            ),
            "preauthored_traffic_visuals": preauthored_traffic_visuals,
            "preauthored_people": preauthored_people,
            "minimum_required_go2_vehicle_clearance_m": (
                args.minimum_go2_vehicle_clearance
                if traffic_manager is not None and not dynamic_agents_ignore_go2
                else None
            ),
            "overview_video": video_validation,
            "road_sweep_video": road_sweep_video_validation,
            "overview_layout": (
                {
                    "kind": "four_panel_2x2",
                    "panels": [
                        ["scene_global", "fixed_local_global"],
                        ["go2_follow", "pedestrian_follow"],
                    ],
                    "panel_resolution": [args.overview_width, args.overview_height],
                    "scene_global_eye_world_xyz": args.scene_global_eye,
                    "scene_global_target_world_xyz": args.scene_global_target,
                    "scene_global_focal_length_mm": args.scene_global_focal_length,
                    "pedestrian_follow_index": args.pedestrian_follow_index,
                }
                if args.joint_four_panel_video
                else
                {
                    "kind": "three_horizontal_panels",
                    "panels": ["fixed_local_global", "go2_follow", "pedestrian_follow"],
                    "panel_resolution": [args.overview_width, args.overview_height],
                    "pedestrian_follow_index": args.pedestrian_follow_index,
                }
                if args.joint_three_panel_video
                else {
                    "kind": "two_horizontal_panels",
                    "panels": [
                        "go2_fixed_oblique"
                        if roaming_payload.get("video", {})
                        .get("go2_follow_camera", {})
                        .get("fixed_eye_world_xyz") is not None
                        else "go2_oblique_follow",
                        "people_fixed_oblique"
                        if roaming_payload.get("video", {})
                        .get("person_follow_camera", {})
                        .get("fixed_eye_world_xyz") is not None
                        else "resident_pedestrian_overhead_follow",
                    ],
                    "panel_resolution": [args.overview_width, args.overview_height],
                    "output_resolution": [args.overview_width * 2, args.overview_height],
                    "pedestrian_follow_index": args.pedestrian_follow_index,
                    "go2_camera": roaming_payload.get("video", {}).get(
                        "go2_follow_camera", {}
                    ),
                    "people_camera": roaming_payload.get("video", {}).get(
                        "person_follow_camera", {}
                    ),
                }
                if args.roaming_ghost_two_panel_video
                else {
                    "kind": "single_panel_moving_world_path",
                    "motion_start_eye_world_xyz": args.overview_motion_start_eye,
                    "motion_end_eye_world_xyz": args.overview_motion_end_eye,
                    "motion_speed_mps": args.overview_motion_speed_mps,
                    "motion_pitch_down_deg": args.overview_motion_pitch_down_deg,
                    "motion_lookahead_m": args.overview_motion_lookahead_m,
                }
                if args.overview_motion_start_eye is not None
                else {"kind": "single_panel"}
            ),
            "evidence": {"trajectory": str(trajectory_path), "trajectory_plot": str(plot_path)},
        }
        if traffic_manager is not None:
            write_json(metadata_dir / "traffic_states.json", traffic_manager.states)
        if three_camera_writer is not None:
            three_camera_writer.close(completed=True)
            three_camera_writer = None
            from urbanverse.dynamic_agents.admission.capture_contract import audit_capture_isolated
            contract = audit_capture_isolated(run_dir)
            write_json(metadata_dir / 'three_camera_contract.json', contract)
            summary['three_camera_contract'] = contract
            summary['three_camera_joint_qualification'] = 'requires_pixel_depth_and_sync_review'
            if contract['status'] == 'failed':
                passed = False
                status = 'failed'
                summary['status'] = status
        write_json(summary_path, summary)
        write_json(
            metadata_dir / "environment.json",
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "hostname": platform.node(),
                "gpu": capture.gpu_snapshot(args.gpu),
                "git_commit": git_commit_at_launch,
                "git_status_short": git_status_at_launch,
                "python_version": sys.version,
                "kit_version": omni.kit.app.get_app().get_build_version(),
                "replicator_extension": omni.kit.app.get_app().get_extension_manager().get_enabled_extension_id("omni.replicator.core"),
                "isaac_sim_version": package_version("isaacsim"),
                "isaac_lab_release": (PROJECT_ROOT / "repos/IsaacLab/VERSION").read_text(encoding="utf-8").strip(),
                "kit_experience": (
                    str(selected_experience) if selected_experience is not None else None
                ),
                "kit_experience_sha256": (
                    capture.sha256(selected_experience)
                    if selected_experience is not None
                    else None
                ),
                "official_task": task,
                "profile": "flat",
                "renderer": (
                    (
                        "RayTracedLighting synchronized scene-global + local-global + "
                        "Go2-follow + pedestrian-follow panels"
                        if args.joint_four_panel_video
                        else
                        "RayTracedLighting synchronized fixed Go2 + fixed official People panels"
                        + (
                            " plus independent road-centre sweep"
                            if road_sweep_payload is not None
                            else ""
                        )
                        if args.roaming_ghost_two_panel_video
                        else
                        "RayTracedLighting synchronized global + Go2-follow + pedestrian-follow panels"
                        if args.joint_three_panel_video
                        else "RayTracedLighting independent constant-speed moving overview"
                        if args.overview_motion_start_eye is not None
                        else "RayTracedLighting fixed world-space oblique global overview"
                        if args.overview_fixed_eye is not None
                        else "RayTracedLighting body-mounted forward pinhole RGB"
                        if args.go2_front_pinhole
                        else "RayTracedLighting elevated third-person chase overview"
                    )
                    if args.overview_video
                    else "none; headless physics-only Scene 10 isolation probe"
                ),
                "overview_video_enabled": bool(args.overview_video),
                "front_pinhole_mount": ({
                    "prim_path": "/World/envs/env_0/Robot/base/FrontPinholeCamera",
                    "position_body_xyz_m": [0.35, 0.0, 0.10],
                    "quaternion_wxyz_world_convention": [1.0, 0.0, 0.0, 0.0],
                    "follows_full_body_attitude": True,
                    "projection": "pinhole", "data_types": ["rgb"],
                } if args.go2_front_pinhole else None),
                "overview_resolution": (
                    [
                        args.overview_width
                        * (
                            2
                            if args.joint_four_panel_video
                            or args.roaming_ghost_two_panel_video
                            else 3
                            if args.joint_three_panel_video
                            else 1
                        ),
                        args.overview_height * (2 if args.joint_four_panel_video else 1),
                    ]
                    if args.overview_video
                    else None
                ),
                "overview_panel_resolution": (
                    [args.overview_width, args.overview_height]
                    if args.overview_video
                    else None
                ),
                "overview_fps": actual_overview_fps if args.overview_video else None,
                "overview_encoding": (
                    {
                        "codec": args.overview_video_codec,
                        "h264_crf": (
                            args.overview_h264_crf
                            if args.overview_video_codec == "h264"
                            else None
                        ),
                        "h264_preset": (
                            args.overview_h264_preset
                            if args.overview_video_codec == "h264"
                            else None
                        ),
                    }
                    if args.overview_video
                    else None
                ),
                "overview_camera": {
                    "policy": (
                        "body-mounted forward pinhole, full base orientation, RGB only"
                        if args.go2_front_pinhole else
                        "synchronized scene-global, fixed-local-global, Go2-follow, and "
                        "pedestrian-follow panels"
                        if args.joint_four_panel_video
                        else
                        "synchronized fixed-global, Go2-follow, and pedestrian-follow panels"
                        if args.joint_three_panel_video
                        else "synchronized fixed Go2 and People panels with an independent road-centre sweep"
                        if args.roaming_ghost_two_panel_video
                        and road_sweep_payload is not None
                        else "synchronized fixed Go2 and People panels"
                        if args.roaming_ghost_two_panel_video
                        else "independent constant-speed world path with fixed pitch"
                        if args.overview_motion_start_eye is not None
                        else "fixed world-space oblique global overview"
                        if args.overview_fixed_eye is not None
                        else (
                            "dynamic-convoy-centroid fixed-offset regression camera"
                            if args.traffic_visibility_diagnostic_convoy
                            else "Go2-relative oblique chase with adjacent-lane coverage"
                        )
                    ),
                    "fixed_eye_world_xyz": args.overview_fixed_eye,
                    "fixed_target_world_xyz": args.overview_fixed_target,
                    "motion_start_eye_world_xyz": args.overview_motion_start_eye,
                    "motion_end_eye_world_xyz": args.overview_motion_end_eye,
                    "motion_speed_mps": args.overview_motion_speed_mps,
                    "motion_pitch_down_deg": args.overview_motion_pitch_down_deg,
                    "motion_lookahead_m": args.overview_motion_lookahead_m,
                    "joint_three_panel_video": bool(args.joint_three_panel_video),
                    "joint_four_panel_video": bool(args.joint_four_panel_video),
                    "roaming_ghost_two_panel_video": bool(args.roaming_ghost_two_panel_video),
                    "scene_global_eye_world_xyz": args.scene_global_eye,
                    "scene_global_target_world_xyz": args.scene_global_target,
                    "scene_global_focal_length_mm": args.scene_global_focal_length,
                    "pedestrian_follow_index": (
                        args.pedestrian_follow_index
                        if joint_people_video
                        else None
                    ),
                    "preview_time_s": args.overview_preview_time_s,
                    "chase_distance_m": args.overview_chase_distance,
                    "lateral_offset_m": args.overview_lateral_offset,
                    "chase_height_m": args.overview_chase_height,
                    "target_forward_m": args.overview_target_forward,
                    "target_lateral_m": args.overview_target_lateral,
                    "target_height_m": 0.35,
                    "focal_length_mm": args.overview_focal_length,
                    "requested_exposure_ev": args.overview_exposure_ev,
                    "applied_exposure_ev": overview_exposure_readback,
                    "output_brightness_factor": args.overview_output_brightness,
                    "output_brightness_scope": "overview visualization only; does not modify sensor or physics state",
                    "motion_blur": overview_motion_blur_readback,
                    "minimum_dynamic_pass_duration_s": args.minimum_visible_vehicle_pass_duration,
                    "runtime_camera_lifecycle": (
                        "USD cameras and Replicator render products created after env.reset; "
                        "each capture flushes Isaac Lab Fabric before Replicator refresh"
                        if args.roaming_ghost_two_panel_video
                        else "Isaac Lab CameraCfg registered before env.reset"
                    ),
                    "two_panel_cameras": (
                        {
                            "go2": roaming_payload.get("video", {}).get(
                                "go2_follow_camera", {}
                            ),
                            "people": roaming_payload.get("video", {}).get(
                                "person_follow_camera", {}
                            ),
                        }
                        if args.roaming_ghost_two_panel_video
                        else None
                    ),
                    "road_sweep_camera": road_sweep_payload,
                } if args.overview_video else None,
                "road_sweep_video": road_sweep_video_validation,
                "fabric_render_bridge": joint_render_bridge_settings,
                "collection_light_scale": light_scale_overrides,
                "source_usd": str(args.source_usd.resolve()),
                "source_usd_sha256": capture.sha256(args.source_usd.resolve()),
                "source_tar": str(args.source_tar.resolve()) if args.source_tar else None,
                "source_tar_sha256": capture.sha256(args.source_tar.resolve()) if args.source_tar else None,
                "reference_route": str(args.reference_route.resolve()),
                "reference_route_sha256": capture.sha256(args.reference_route.resolve()),
                "policy_sha256": capture.sha256(args.policy.resolve()),
                "checkpoint_sha256": capture.sha256(args.checkpoint.resolve()),
                "policy_kind": args.policy_kind,
                "actor": actor_metadata,
                "scene10_multivehicle_traffic": bool(traffic_manager is not None),
                "animated_pedestrians": bool(
                    people_manager is not None or roaming_people_manager is not None
                ),
                "official_resident_people": bool(roaming_people_manager is not None),
                "project_resident_people": bool(mixed_roaming_payload is not None),
                "roaming_ghost_config": (
                    str(roaming_ghost_config_path) if roaming_ghost_config_path else None
                ),
                "roaming_ghost_config_sha256": (
                    capture.sha256(roaming_ghost_config_path)
                    if roaming_ghost_config_path
                    else None
                ),
                "mixed_roaming_config": (
                    str(mixed_roaming_config_path) if mixed_roaming_config_path else None
                ),
                "mixed_roaming_config_sha256": (
                    roaming_config_sha256_at_load
                    if mixed_roaming_config_path
                    else None
                ),
                "ghost_ab_mode": (
                    "normal_collision"
                    if args.ghost_ab_normal and roaming_ghost_config_path
                    else "ghost_passthrough"
                    if roaming_ghost_config_path
                    else None
                ),
                "ghost_ab_headless": bool(args.ghost_ab_headless),
                "preauthored_traffic_visuals": preauthored_traffic_visuals,
                "preauthored_people": preauthored_people,
                "pedestrian_config": (
                    str(args.pedestrian_config.resolve()) if args.pedestrian_config else None
                ),
                "pedestrian_config_sha256": (
                    capture.sha256(args.pedestrian_config.resolve())
                    if args.pedestrian_config
                    else None
                ),
                "traffic_registry": str(args.traffic_registry.resolve()) if args.traffic_registry else None,
                "traffic_registry_sha256": capture.sha256(args.traffic_registry.resolve()) if args.traffic_registry else None,
                "traffic_audit_inventory": str(args.traffic_audit_inventory.resolve()) if args.traffic_audit_inventory else None,
                "traffic_audit_inventory_sha256": capture.sha256(args.traffic_audit_inventory.resolve()) if args.traffic_audit_inventory else None,
                "traffic_validated_routes": str(args.traffic_validated_routes.resolve()) if args.traffic_validated_routes else None,
                "traffic_validated_routes_sha256": capture.sha256(args.traffic_validated_routes.resolve()) if args.traffic_validated_routes else None,
                "traffic_validated_static_bodies": str(args.traffic_validated_static_bodies.resolve()) if args.traffic_validated_static_bodies else None,
                "traffic_validated_static_bodies_sha256": capture.sha256(args.traffic_validated_static_bodies.resolve()) if args.traffic_validated_static_bodies else None,
                "traffic_continuous_looping": bool(args.traffic_continuous_looping),
                "traffic_vehicle_count": int(args.traffic_vehicle_count),
                "traffic_initial_fill": bool(args.traffic_initial_fill),
                "traffic_automotive_routes": str(args.traffic_automotive_routes.resolve()) if args.traffic_automotive_routes else None,
                "traffic_automotive_routes_sha256": capture.sha256(args.traffic_automotive_routes.resolve()) if args.traffic_automotive_routes else None,
                "command_line": [sys.executable, *sys.argv],
                "step_dt_s": dt,
                "locomotion_training_performed": False,
                "go2_highlight_material": go2_highlight_metadata,
            },
        )
        print(f"RESULT summary={summary_path} status={status} stop_reason={stop_reason}", flush=True)
        # A completed process is not necessarily an accepted experiment.  The
        # provenance-aware runners use the exit status as a hard phase gate;
        # returning zero for a written-but-failed summary could otherwise let
        # technical or formal phases proceed on invalid physics evidence.
        result_code = 0 if passed else 2
    except Exception as exc:
        traceback.print_exc()
        write_json(
            summary_path,
            {
                "status": "failed",
                "mode": args.mode,
                "failure_stage": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "end_to_end_wall_s": time.perf_counter() - started,
            },
        )
        result_code = 1
    finally:
        if three_camera_writer is not None:
            three_camera_writer.close()
        if overlap_recorder is not None:
            overlap_recorder.close(completed=False)
        if trajectory_handle is not None:
            trajectory_handle.close()
        if video_writer is not None:
            video_writer.release()
        if road_sweep_video_writer is not None:
            road_sweep_video_writer.release()
        if "official_people_runtime" in locals() and official_people_runtime is not None:
            try:
                official_people_runtime.stop()
            except Exception:
                traceback.print_exc()
        if env is not None:
            try:
                if args.go2_three_camera:
                    from urbanverse.dynamic_agents.rendering.camera_teardown import detach_camera_outputs
                    teardown = detach_camera_outputs(env.unwrapped.scene)
                    write_json(metadata_dir / 'camera_teardown.json', teardown)
                env.close()
            except Exception:
                traceback.print_exc()
                if args.go2_three_camera:
                    result_code = 1
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                traceback.print_exc()
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
