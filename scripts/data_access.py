from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:"
            f"\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
        )


def ensure_dvc_data(
    *,
    data_path: str,
    remote_name: str,
    timeout_seconds: int,
    jobs: int,
    repo_root: Path | None = None,
) -> None:
    repo_root = repo_root or Path.cwd()

    token = os.getenv("YANDEX_WEBDAV_TOKEN")
    if not token:
        raise RuntimeError(
            "YANDEX_WEBDAV_TOKEN is not set.\n"
            "macOS/Linux: export YANDEX_WEBDAV_TOKEN='...'\n"
            "PowerShell:  $env:YANDEX_WEBDAV_TOKEN='...'\n"
        )

    _run(
        ["dvc", "remote", "modify", "--local", remote_name, "password", token],
        repo_root,
    )

    _run(
        ["dvc", "remote", "modify", remote_name, "timeout", str(timeout_seconds)],
        repo_root,
    )

    _run(["dvc", "pull", data_path, "-j", str(jobs)], repo_root)
