"""Run manifest: what was run, on what, with which settings, and how long it took.

Usage in a notebook:
    from benchbias.runlog import RunLog
    log = RunLog("01_bbq_rq1", config={"encoder": ENCODER, "min_cluster": 10, "seed": SEED})
    with log.step("embed"):
        ...
    log.add_result("AMI_masked", 0.41)
    log.save("outputs/")      # -> outputs/run_01_bbq_rq1_<timestamp>.json
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "sentence-transformers", "transformers",
            "umap-learn", "hdbscan", "bertopic", "datasets"]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5, cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except Exception:
        return None


def hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "colab": "google.colab" in sys.modules}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        info["cuda"] = None
    return info


def versions(packages=PACKAGES) -> dict:
    out = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


class RunLog:
    def __init__(self, name: str, config: dict | None = None):
        self.name = name
        self.started = datetime.now(timezone.utc)
        self.data = {"run": name, "started_utc": self.started.isoformat(timespec="seconds"),
                     "git_commit": _git_commit(), "config": dict(config or {}),
                     "hardware": hardware(), "versions": versions(), "timings_s": {}, "results": {}}

    @contextlib.contextmanager
    def step(self, label: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.data["timings_s"][label] = round(time.perf_counter() - t0, 2)

    def add_result(self, key: str, value) -> None:
        self.data["results"][key] = value

    def save(self, out_dir: str | Path = "outputs") -> Path:
        self.data["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.data["total_s"] = round((datetime.now(timezone.utc) - self.started).total_seconds(), 1)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"run_{self.name}_{self.started.strftime('%Y%m%dT%H%M%S')}.json"
        path.write_text(json.dumps(self.data, indent=2, default=str))
        return path
