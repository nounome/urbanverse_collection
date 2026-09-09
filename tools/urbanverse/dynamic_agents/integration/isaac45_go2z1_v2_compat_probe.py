#!/usr/bin/env python3
"""Quick compatibility probe for the third-party Go2+Z1 V2 locomotion actor.

The downloaded policy observes 18 joint positions and velocities but controls
only the 12 Go2 leg joints.  This probe runs it on the stock pure-Go2 flat task
by inserting six zero arm-position errors and six zero arm velocities into the
stock 48-value Go2 observation.  It is an approximation, not a claim of native
policy compatibility: the Z1 arm mass and inertia are absent from the robot.
"""

from __future__ import annotations

import argparse
import json
import math
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


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from urbanverse.dynamic_agents.navigation import control as navigation  # noqa: E402
from urbanverse.dynamic_agents.integration import go2_route_capture as route_capture  # noqa: E402
from urbanverse.viz_style import load_font  # noqa: E402


CASES = (
    {"label": "cold_wz_pos_0p5", "pre_vx": 0.0, "pivot_wz": 0.5},
    {"label": "cold_wz_pos_1p0", "pre_vx": 0.0, "pivot_wz": 1.0},
    {"label": "forward_then_wz_pos_0p5", "pre_vx": 0.30, "pivot_wz": 0.5},
    {"label": "forward_then_wz_pos_1p0", "pre_vx": 0.30, "pivot_wz": 1.0},
    {"label": "forward_then_wz_neg_0p5", "pre_vx": 0.30, "pivot_wz": -0.5},
    {"label": "forward_then_wz_neg_1p0", "pre_vx": 0.30, "pivot_wz": -1.0},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--precondition-s", type=float, default=8.0)
    parser.add_argument("--pivot-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def flat_ground_usda() -> str:
    return """#usda 1.0
(
    defaultPrim = "Ground"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Ground"
{
    def Mesh "Surface" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {
        uniform bool doubleSided = 1
        bool physics:collisionEnabled = 1
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1000, -1000, 0), (1000, -1000, 0), (1000, 1000, 0), (-1000, 1000, 0)]
        uniform token subdivisionScheme = "none"
    }
}
"""


def build_actor(torch, checkpoint: Path, device: str):
    payload = torch.load(str(checkpoint), map_location=device, weights_only=True)
    state = payload["actor_state_dict"]
    shapes = {key: list(value.shape) for key, value in state.items() if key.startswith("mlp.")}
    input_dim = int(state["mlp.0.weight"].shape[1])
    hidden = [int(state[f"mlp.{index}.weight"].shape[0]) for index in (0, 2, 4)]
    output_dim = int(state["mlp.6.weight"].shape[0])
    if input_dim != 60 or output_dim != 12 or len(set(hidden)) != 1:
        raise ValueError(f"unexpected actor shape: input={input_dim}, hidden={hidden}, output={output_dim}")
    actor = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden[0]),
        torch.nn.ELU(),
        torch.nn.Linear(hidden[0], hidden[1]),
        torch.nn.ELU(),
        torch.nn.Linear(hidden[1], hidden[2]),
        torch.nn.ELU(),
        torch.nn.Linear(hidden[2], output_dim),
    ).to(device).eval()
    actor.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")})
    return actor, {"checkpoint_iteration": payload.get("iter"), "tensor_shapes": shapes}


def adapt_pure_go2_observation(torch, observation):
    if observation.shape[-1] != 48:
        raise ValueError(f"expected stock Go2 48-D observation, got {tuple(observation.shape)}")
    zeros = torch.zeros((*observation.shape[:-1], 6), dtype=observation.dtype, device=observation.device)
    # Stock Go2: [base/command 12, leg_pos 12, leg_vel 12, last_action 12].
    # Go2+Z1:    [base/command 12, all_pos 18, all_vel 18, leg_action 12].
    return torch.cat(
        (observation[..., :24], zeros, observation[..., 24:36], zeros, observation[..., 36:48]), dim=-1
    )


def wrap_delta(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return (current - previous + np.pi) % (2.0 * np.pi) - np.pi


def draw_diagnostic(path: Path, results: list[dict[str, Any]], paths: list[np.ndarray]) -> None:
    width, height = 1300, 760
    image = Image.new("RGB", (width, height), (239, 244, 247))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "Go2+Z1 V2 → 纯 Go2 快速兼容测试", fill=(20, 29, 36), font=load_font(30, bold=True))
    draw.text((24, 60), "先静止/直行 8 秒，再原地转向 8 秒；轨迹已换算到各自起点", fill=(55, 70, 80), font=load_font(18))
    colors = {"passed": (45, 158, 91), "failed": (198, 62, 62)}
    cell_w, cell_h = 410, 310
    for index, (row, points) in enumerate(zip(results, paths)):
        column, line = index % 3, index // 3
        x0, y0 = 24 + column * 425, 105 + line * 325
        draw.rounded_rectangle((x0, y0, x0 + cell_w, y0 + cell_h), radius=12, fill=(255, 255, 255), outline=(160, 174, 182), width=2)
        draw.text((x0 + 14, y0 + 12), row["label"], fill=colors[row["status"]], font=load_font(19, bold=True))
        plot_box = (x0 + 18, y0 + 52, x0 + 392, y0 + 225)
        draw.rectangle(plot_box, fill=(16, 29, 37))
        if len(points) > 1:
            low, high = points.min(axis=0), points.max(axis=0)
            span = np.maximum(high - low, 0.5)
            scale = min((plot_box[2] - plot_box[0] - 20) / span[0], (plot_box[3] - plot_box[1] - 20) / span[1])
            projected = [
                (plot_box[0] + 10 + (p[0] - low[0]) * scale, plot_box[3] - 10 - (p[1] - low[1]) * scale)
                for p in points
            ]
            draw.line(projected, fill=(242, 151, 55), width=4)
            draw.ellipse((projected[0][0] - 5, projected[0][1] - 5, projected[0][0] + 5, projected[0][1] + 5), fill=(65, 150, 240))
            draw.ellipse((projected[-1][0] - 5, projected[-1][1] - 5, projected[-1][0] + 5, projected[-1][1] + 5), fill=(230, 70, 70))
        lines = (
            f"转向累计: {row['pivot_signed_yaw_rad']:+.2f} rad",
            f"末段转速: {row['pivot_tail_signed_yaw_rate_radps']:+.2f} rad/s",
            f"最大漂移半径: {row['pivot_maximum_radius_m']:.2f} m  重置: {row['termination_count']}",
        )
        for offset, text in enumerate(lines):
            draw.text((x0 + 14, y0 + 236 + offset * 22), text, fill=(40, 52, 60), font=load_font(15))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    metadata_dir = run_dir / "metadata"
    captures_dir = run_dir / "captures"
    visualizations_dir = run_dir / "visualizations"
    wrapper_dir = run_dir / "wrapper"
    for directory in (metadata_dir, captures_dir, visualizations_dir, wrapper_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary_path = metadata_dir / "summary.json"
    write_json(summary_path, {"status": "running", "case_count": len(CASES)})
    wrapper_path = wrapper_dir / "flat_ground.usda"
    wrapper_path.write_text(flat_ground_usda(), encoding="utf-8")
    simulation_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=False, device=f"cuda:{args.gpu}", multi_gpu=False)
        simulation_app = launcher.app

        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        task = navigation.CONTROLLER_PROFILES["flat"]["task"]
        cfg = parse_env_cfg(task, device=f"cuda:{args.gpu}", num_envs=len(CASES))
        route_capture.configure_environment(
            cfg,
            wrapper_path,
            {"spawn_position": [0.0, 0.0, 0.4], "spawn_yaw_rad": 0.0},
            None,
            num_envs=len(CASES),
        )
        cfg.scene.env_spacing = 4.0
        cfg.episode_length_s = args.settle_s + args.precondition_s + args.pivot_s + 10.0
        cfg.seed = args.seed
        env = gym.make(task, cfg=cfg)
        observations, _ = env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene["contact_forces"]
        dt = float(unwrapped.step_dt)
        actor, actor_metadata = build_actor(torch, args.checkpoint.resolve(), f"cuda:{args.gpu}")
        command_term = unwrapped.command_manager.get_term("base_velocity")
        command_term.time_left[:] = 1000.0
        command_term.is_heading_env[:] = False
        command_term.is_standing_env[:] = False

        def step_with(command):
            nonlocal observations
            command_term.vel_command_b[:] = command
            obs = observations["policy"] if isinstance(observations, dict) else observations
            actions = actor(adapt_pure_go2_observation(torch, obs))
            observations, _, terminated, truncated, _ = env.step(actions)
            return terminated, truncated

        zero = torch.zeros((len(CASES), 3), dtype=torch.float32, device=unwrapped.device)
        with torch.inference_mode():
            for _ in range(int(round(args.settle_s / dt))):
                step_with(zero)

        initial = robot.data.root_pos_w.detach().cpu().numpy().astype(np.float64)
        paths = [[position[:2].copy()] for position in initial]
        minimum_height = initial[:, 2].copy()
        termination_count = np.zeros(len(CASES), dtype=np.int64)
        nonfoot_steps = np.zeros(len(CASES), dtype=np.int64)
        contact_names = list(contact_sensor.body_names)
        nonfoot_indices = [index for index, name in enumerate(contact_names) if not name.endswith("_foot")]
        pre_command = torch.tensor(
            [[case["pre_vx"], 0.0, 0.0] for case in CASES], dtype=torch.float32, device=unwrapped.device
        )
        pivot_command = torch.tensor(
            [[0.0, 0.0, case["pivot_wz"]] for case in CASES], dtype=torch.float32, device=unwrapped.device
        )
        pivot_yaw_accum = np.zeros(len(CASES), dtype=np.float64)
        pivot_path_length = np.zeros(len(CASES), dtype=np.float64)
        pivot_maximum_radius = np.zeros(len(CASES), dtype=np.float64)
        pivot_start_positions = None
        pivot_previous_positions = None
        pivot_rates: list[np.ndarray] = []
        previous_yaw = None
        run_started = time.perf_counter()
        phases = (("precondition", args.precondition_s, pre_command), ("pivot", args.pivot_s, pivot_command))
        with torch.inference_mode():
            for phase, duration, command in phases:
                if phase == "pivot":
                    pivot_start_positions = robot.data.root_pos_w.detach().cpu().numpy().astype(np.float64)
                    pivot_previous_positions = pivot_start_positions.copy()
                for step_index in range(int(round(duration / dt))):
                    terminated, truncated = step_with(command)
                    positions = robot.data.root_pos_w.detach().cpu().numpy().astype(np.float64)
                    yaw = np.asarray(
                        [route_capture.yaw_from_quat(value) for value in robot.data.root_quat_w.detach().cpu().numpy()],
                        dtype=np.float64,
                    )
                    rates = robot.data.root_ang_vel_b[:, 2].detach().cpu().numpy().astype(np.float64)
                    minimum_height = np.minimum(minimum_height, positions[:, 2])
                    termination_count += np.logical_or(
                        terminated.detach().cpu().numpy(), truncated.detach().cpu().numpy()
                    ).astype(np.int64)
                    contact = contact_sensor.data.net_forces_w.detach().cpu().numpy().astype(np.float64)
                    if nonfoot_indices:
                        nonfoot_steps += np.any(np.linalg.norm(contact[:, nonfoot_indices], axis=-1) > 5.0, axis=1)
                    for index, position in enumerate(positions):
                        paths[index].append(position[:2].copy())
                    if phase == "pivot":
                        if previous_yaw is not None:
                            pivot_yaw_accum += wrap_delta(yaw, previous_yaw)
                        pivot_path_length += np.linalg.norm(
                            positions[:, :2] - pivot_previous_positions[:, :2], axis=1
                        )
                        pivot_maximum_radius = np.maximum(
                            pivot_maximum_radius,
                            np.linalg.norm(positions[:, :2] - pivot_start_positions[:, :2], axis=1),
                        )
                        pivot_previous_positions = positions.copy()
                        pivot_rates.append(rates.copy())
                    previous_yaw = yaw
        simulation_wall_s = time.perf_counter() - run_started
        tail_steps = max(1, int(round(min(3.0, args.pivot_s) / dt)))
        tail_rates = np.stack(pivot_rates[-tail_steps:])
        results = []
        relative_paths = []
        for index, case in enumerate(CASES):
            sign = 1.0 if case["pivot_wz"] > 0 else -1.0
            signed_yaw = float(pivot_yaw_accum[index] * sign)
            signed_rate = float(np.median(tail_rates[:, index]) * sign)
            final_position = np.asarray(paths[index][-1], dtype=np.float64)
            final_displacement = float(
                np.linalg.norm(final_position - pivot_start_positions[index, :2])
            )
            stable = bool(termination_count[index] == 0 and nonfoot_steps[index] <= int(round(0.1 / dt)))
            responsive = bool(signed_yaw >= 1.0 and signed_rate >= 0.10)
            row = {
                **case,
                "status": "passed" if stable and responsive else "failed",
                "pivot_signed_yaw_rad": signed_yaw,
                "pivot_tail_signed_yaw_rate_radps": signed_rate,
                "pivot_path_length_m": float(pivot_path_length[index]),
                "pivot_final_displacement_m": final_displacement,
                "pivot_maximum_radius_m": float(pivot_maximum_radius[index]),
                "minimum_base_height_m": float(minimum_height[index]),
                "termination_count": int(termination_count[index]),
                "nonfoot_contact_steps": int(nonfoot_steps[index]),
                "stable": stable,
                "rotation_responsive": responsive,
            }
            results.append(row)
            points = np.asarray(paths[index], dtype=np.float64)
            relative_paths.append(points - points[0])
        write_json(metadata_dir / "results.json", results)
        diagnostic = visualizations_dir / "go2z1_v2_compatibility.png"
        draw_diagnostic(diagnostic, results, relative_paths)
        Image.open(diagnostic).convert("RGB").save(captures_dir / "compatibility_diagnostic_rgb.png")
        passed = sum(row["status"] == "passed" for row in results)
        transition_passed = sum(
            row["status"] == "passed" for row in results if row["label"].startswith("forward_then")
        )
        summary = {
            "status": "success",
            "assessment": "promising" if passed == len(results) else "partially_compatible" if passed else "not_compatible",
            "case_count": len(results),
            "passed_case_count": passed,
            "forward_to_pivot_passed_count": transition_passed,
            "simulation_duration_s": args.settle_s + args.precondition_s + args.pivot_s,
            "simulation_wall_s": simulation_wall_s,
            "end_to_end_wall_s": time.perf_counter() - started,
            "adapter": "48-D pure-Go2 observation to 60-D Go2+Z1 observation by zero-padding folded-arm relative position/velocity",
            "compatibility_limit": "Pure Go2 lacks the Z1 arm mass/inertia used during training; this is an empirical approximation.",
            "actor": actor_metadata,
            "evidence": {"results": str(metadata_dir / "results.json"), "diagnostic": str(diagnostic)},
        }
        write_json(summary_path, summary)
        write_json(
            metadata_dir / "environment.json",
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "hostname": platform.node(),
                "gpu_index": args.gpu,
                "gpu": route_capture.gpu_snapshot(args.gpu),
                "driver": run_text(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                "isaac_sim_version": package_version("isaacsim"),
                "isaac_lab_release": (PROJECT_ROOT / "repos/IsaacLab/VERSION").read_text(encoding="utf-8").strip(),
                "python_version": sys.version.replace("\n", " "),
                "git_commit": run_text(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]),
                "git_status_short": run_text(["git", "-C", str(PROJECT_ROOT), "status", "--short"]),
                "command_line": [sys.executable, *sys.argv],
                "renderer": "none; headless physics-only flat-ground compatibility probe",
                "resolution": None,
                "camera": None,
                "official_task": task,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": route_capture.sha256(args.checkpoint.resolve()),
                "checkpoint_source": "https://huggingface.co/m3/go2z1-walking-rsl-rl-v2",
                "checkpoint_revision": "e7ae856642e0a1c74427bb00578f1a817318ee8f",
                "step_dt_s": dt,
                "ground_usd": str(wrapper_path),
                "ground_usd_sha256": route_capture.sha256(wrapper_path),
                "locomotion_training_performed": False,
            },
        )
        print(f"RESULT summary={summary_path} assessment={summary['assessment']}", flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_json(
            summary_path,
            {
                "status": "failed",
                "failure_stage": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "end_to_end_wall_s": time.perf_counter() - started,
            },
        )
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
