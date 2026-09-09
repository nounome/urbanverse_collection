#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "Oatmealliu/UrbanVerse-CraftBench"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=Path("data/urbanverse_craftbench/raw"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_dir = args.run_dir.resolve()
    local_dir = args.local_dir.resolve()
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "download_manifest.json"

    api = HfApi()
    info = api.dataset_info(args.repo_id)
    files = sorted(api.list_repo_files(args.repo_id, repo_type="dataset"))

    manifest: dict[str, Any] = {
        "status": "running",
        "repo_id": args.repo_id,
        "local_dir": str(local_dir),
        "run_dir": str(run_dir),
        "started_at": started_at,
        "file_count": len(files),
        "siblings_count": len(info.siblings or []),
        "files": [],
    }
    write_json(manifest_path, manifest)

    print(f"[craftbench-download] started_at={started_at}", flush=True)
    print(f"[craftbench-download] repo_id={args.repo_id}", flush=True)
    print(f"[craftbench-download] local_dir={local_dir}", flush=True)
    print(f"[craftbench-download] file_count={len(files)}", flush=True)

    ok_count = 0
    failed: list[dict[str, Any]] = []
    total_downloaded_size = 0

    for idx, filename in enumerate(files, start=1):
        rel_path = Path(filename)
        dst = local_dir / rel_path
        before_size = file_size(dst)
        print(f"[craftbench-download] {idx}/{len(files)} {filename}", flush=True)
        item: dict[str, Any] = {
            "filename": filename,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "before_size": before_size,
            "status": "failed",
            "attempts": 0,
        }

        last_error = None
        for attempt in range(1, max(1, args.max_retries) + 1):
            item["attempts"] = attempt
            try:
                downloaded_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    force_download=False,
                )
                downloaded_path = Path(downloaded_path)
                final_size = file_size(downloaded_path)
                item.update(
                    {
                        "status": "success",
                        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "path": str(downloaded_path),
                        "size": final_size,
                    }
                )
                ok_count += 1
                if final_size:
                    total_downloaded_size += final_size
                print(f"[craftbench-download] ok {filename} size={final_size}", flush=True)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                item["last_error"] = last_error
                print(f"[craftbench-download] attempt {attempt} failed for {filename}: {last_error}", flush=True)
                if attempt < args.max_retries:
                    time.sleep(min(60, 5 * attempt))
        else:
            failed.append({"filename": filename, "error": last_error})
            print(f"[craftbench-download] failed {filename}", flush=True)

        manifest["files"].append(item)
        manifest.update(
            {
                "status": "running",
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "ok_count": ok_count,
                "failed_count": len(failed),
                "total_recorded_size": total_downloaded_size,
            }
        )
        write_json(manifest_path, manifest)

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest.update(
        {
            "status": "success" if not failed else "failed",
            "finished_at": finished_at,
            "ok_count": ok_count,
            "failed_count": len(failed),
            "failed": failed,
            "total_recorded_size": total_downloaded_size,
        }
    )
    write_json(manifest_path, manifest)
    print(f"[craftbench-download] finished_at={finished_at}", flush=True)
    print(f"[craftbench-download] ok_count={ok_count} failed_count={len(failed)}", flush=True)
    print(f"[craftbench-download] manifest={manifest_path}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
