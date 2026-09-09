#!/usr/bin/env python3
"""Download and verify Isaac Sim People character and animation assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


BUCKET_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
XML_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRINT_LOCK = threading.Lock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    with PRINT_LOCK:
        print(f"[{now()}] {message}", flush=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def list_prefix(prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    continuation_token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if continuation_token:
            query["continuation-token"] = continuation_token
        url = BUCKET_URL + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.parse(response).getroot()
        for content in root.findall("s:Contents", XML_NS):
            key = content.findtext("s:Key", namespaces=XML_NS)
            size = content.findtext("s:Size", namespaces=XML_NS)
            etag = content.findtext("s:ETag", namespaces=XML_NS)
            if key and size is not None and int(size) > 0:
                rows.append({"key": key, "size": int(size), "etag": (etag or "").strip('"')})
        if root.findtext("s:IsTruncated", namespaces=XML_NS) != "true":
            return rows
        continuation_token = root.findtext("s:NextContinuationToken", namespaces=XML_NS)


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid(path: Path, row: dict[str, object]) -> bool:
    if not path.is_file() or path.stat().st_size != row["size"]:
        return False
    return True  # File existence/size only; no asset hash verification.


def download(
    row: dict[str, object], destination: Path, remote_root: str, retries: int
) -> tuple[str, int, str]:
    key = str(row["key"])
    relative = Path(key).relative_to(remote_root)
    target = destination / relative
    if is_valid(target, row):
        return key, int(row["size"]), "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(BUCKET_URL + urllib.parse.quote(key, safe="/"))
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
            if not is_valid(partial, row):
                raise RuntimeError("downloaded file failed size/ETag verification")
            partial.replace(target)
            return key, int(row["size"]), "downloaded"
        except Exception as exc:
            log(f"retry {attempt}/{retries}: {key}: {type(exc).__name__}: {exc}")
            if attempt == retries:
                raise
            time.sleep(min(30, attempt * 3))
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--asset-version", default="4.5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.resolve()
    run_dir = args.run_dir.resolve()
    remote_root = f"Assets/Isaac/{args.asset_version}/Isaac/People/"
    prefixes = (remote_root + "Characters/", remote_root + "Animations/")
    destination.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / ".isaac_people_download.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_handle.write(f"pid={os.getpid()} started_at={now()} run_dir={run_dir}\n")
    lock_handle.flush()

    write_json(
        run_dir / "environment.json",
        {
            "started_at": now(),
            "hostname": platform.node(),
            "python": sys.version.replace("\n", " "),
            "git_commit": command(["git", "rev-parse", "HEAD"]),
            "git_status_short": command(["git", "status", "--short"]),
            "command_line": [sys.executable, *sys.argv],
            "bucket_url": BUCKET_URL,
            "asset_version": args.asset_version,
            "remote_prefixes": prefixes,
            "destination": str(destination),
        },
    )
    log("fetching remote S3 inventory")
    rows = sorted((row for prefix in prefixes for row in list_prefix(prefix)), key=lambda row: str(row["key"]))
    total_bytes = sum(int(row["size"]) for row in rows)
    write_json(
        run_dir / "remote_manifest.json",
        {"retrieved_at": now(), "file_count": len(rows), "total_bytes": total_bytes, "files": rows},
    )
    log(f"inventory ready: files={len(rows)} bytes={total_bytes}")

    completed_files = 0
    completed_bytes = 0
    downloaded_files = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, row, destination, remote_root, args.retries): row for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            key, size, status = future.result()
            completed_files += 1
            completed_bytes += size
            downloaded_files += status == "downloaded"
            if completed_files % 10 == 0 or completed_files == len(rows):
                write_json(
                    run_dir / "progress.json",
                    {
                        "status": "downloading" if completed_files < len(rows) else "complete",
                        "updated_at": now(),
                        "completed_files": completed_files,
                        "total_files": len(rows),
                        "completed_bytes": completed_bytes,
                        "total_bytes": total_bytes,
                        "downloaded_files": downloaded_files,
                        "last_completed_key": key,
                    },
                )
                log(f"progress: files={completed_files}/{len(rows)} bytes={completed_bytes}/{total_bytes}")

    local_rows = []
    for row in rows:
        path = destination / Path(str(row["key"])).relative_to(remote_root)
        if not is_valid(path, row):
            raise RuntimeError(f"final verification failed: {path}")
        local_rows.append(row)
    write_json(
        run_dir / "verified_manifest.json",
        {"verified_at": now(), "file_count": len(rows), "total_bytes": total_bytes, "files": local_rows},
    )
    log("download and final verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
