#!/usr/bin/env python3
"""Convert calibrated GLB vehicle visuals to reusable USD layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--asset-count", type=int, default=8)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    converted_dir = run_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=f"cuda:{args.gpu}", multi_gpu=False)
    app = launcher.app
    rows = []
    status = "failed"
    try:
        from isaaclab.sim.converters.mesh_converter import MeshConverter
        from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg

        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        by_id = {row["asset_id"]: row for row in catalog["records"]}
        selected_ids = catalog.get("default_asset_ids", list(by_id))[: args.asset_count]
        for asset_id in selected_ids:
            record = by_id[asset_id]
            source = Path(record["source_asset"]).resolve()
            target = converted_dir / record["asset_id"] / "vehicle.usd"
            target.parent.mkdir(parents=True, exist_ok=True)
            before = time.perf_counter()
            converter = MeshConverter(
                MeshConverterCfg(
                    asset_path=str(source),
                    usd_dir=str(target.parent),
                    usd_file_name=target.name,
                    force_usd_conversion=True,
                    make_instanceable=False,
                    # UrbanVerse vehicle GLBs arrive with their height on Y
                    # and longitudinal axis on Z after asset conversion.  A
                    # +90 degree X rotation standardizes them to Z-up with the
                    # vehicle front along -Y.  The traffic manager then only
                    # needs to apply road yaw around Z.
                    rotation=(
                        math.cos(math.pi / 4.0),
                        math.sin(math.pi / 4.0),
                        0.0,
                        0.0,
                    ),
                )
            )
            target = Path(converter.usd_path)
            if not target.is_file():
                raise RuntimeError(f"converter reported success without output: {target}")
            rows.append(
                {
                    "asset_id": record["asset_id"],
                    "source_glb": str(source),
                    "source_glb_sha256": sha256(source),
                    "converted_usd": str(target),
                    "converted_usd_sha256": sha256(target),
                    "wall_s": time.perf_counter() - before,
                }
            )
            print(f"CONVERTED {record['asset_id']} -> {target}", flush=True)
        status = "success"
        return 0
    finally:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        write_json(
            run_dir / "metadata" / "summary.json",
            {"status": status, "asset_count": len(rows), "assets": rows, "wall_s": time.perf_counter() - started},
        )
        write_json(
            run_dir / "metadata" / "environment.json",
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "hostname": platform.node(),
                "gpu_index": args.gpu,
                "git_commit": commit,
                "catalog": str(args.catalog.resolve()),
                "catalog_sha256": sha256(args.catalog.resolve()),
                "isaac_sim": "4.5.0.0",
                "operation": "omni.kit.asset_converter GLB to USD with materials",
            },
        )
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
