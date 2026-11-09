"""Turn SciNet++ stage artefacts into one machine-readable ``result.json``.

We deliberately do NOT parse console text. Existing scripts already persist
numeric results (``evaluate.py`` -> ``horizon_metrics.npy``) and a checkpoint
(``train.py`` -> ``checkpoint.pt``); we read those and nothing else. This keeps
the Agent/Runtime contract stable even if log wording changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_horizon_metrics(out_dir: str | Path) -> dict[str, float]:
    """``horizon_metrics.npy`` rows are ``[horizon, latent_mse, cosine]``."""
    import numpy as np

    path = Path(out_dir) / "horizon_metrics.npy"
    if not path.is_file():
        return {}
    arr = np.asarray(np.load(path, allow_pickle=False))
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"unexpected horizon_metrics.npy shape: {arr.shape}")
    metrics: dict[str, float] = {}
    for row in arr:
        k = int(row[0])
        metrics[f"latent_mse@{k}"] = float(row[1])
        metrics[f"cosine@{k}"] = float(row[2])
    return metrics


def describe_checkpoint(out_dir: str | Path) -> dict | None:
    path = Path(out_dir) / "checkpoint.pt"
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def collect_artifacts(out_dir: str | Path) -> list[str]:
    """Top-level files produced by the stage (relative names, sorted)."""
    root = Path(out_dir)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_file() and not (
            p.name == "result.json"
            or (p.name.startswith("result_") and p.suffix == ".json")
        )
    )


def log_tail(log_path: str | Path | None, max_chars: int = 3000) -> str:
    if not log_path or not Path(log_path).is_file():
        return ""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def build_result(*, job: dict, status: str, exit_code: int | None,
                 out_dir: str | Path, engine: str = "scinet++",
                 started_at: str | None = None, finished_at: str | None = None,
                 error: str | None = None, error_type: str | None = None,
                 dataset_version: str | None = None) -> dict[str, Any]:
    """Single shape for success AND failure -- the Runtime never sees a partial dict."""
    out = Path(out_dir)
    metrics = parse_horizon_metrics(out) if status == "succeeded" else {}
    checkpoint = describe_checkpoint(out) if status == "succeeded" else None
    spec = job.get("spec") or {}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": job.get("experiment_id"),
        "job_id": job.get("job_id"),
        "stage": job.get("stage"),
        "status": status,
        "engine": engine,
        "run_id": job.get("run_id"),
        "model_version": spec.get("model_version"),
        "dataset_version": dataset_version or spec.get("dataset_version"),
        "metrics": metrics,
        "checkpoint": checkpoint,
        "artifacts": collect_artifacts(out),
        "log_path": job.get("log_path"),
        "result_path": job.get("result_path"),
        "exit_code": exit_code,
        "error_type": error_type,
        "error": error,
        "started_at": started_at or job.get("started_at"),
        "finished_at": finished_at,
    }


def write_result(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                              sort_keys=True), encoding="utf-8")
    tmp.replace(p)  # atomic: readers never see a half-written result.json


def read_result(path: str | Path | None) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
