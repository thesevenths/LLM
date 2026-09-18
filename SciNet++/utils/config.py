"""
utils.config
============
SciNet++ (2026) -- config loading and device/output resolution.

Every stage script (train / analyze / symbolic / evaluate / tta) shares these
helpers so the YAML config is the single source of truth and CUDA falls back to
CPU automatically when unavailable.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys

import torch
import yaml


def make_run_id() -> str:
    """A sortable timestamp used as the per-run subdirectory name.

    Format ``YYYYMMDD_HHMMSS`` is lexicographically orderable, so the "most
    recent run" for an experiment is simply the last entry when the directory
    is sorted.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(path: str = "configs/pendulum.yaml") -> dict:
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(cfg: dict) -> torch.device:
    """Return a torch.device, honouring cfg['device'] but falling back to CPU.

    Prints an explicit log line so the user can verify which device is active.
    If CUDA was requested but is unavailable, a warning is printed explaining
    why (missing driver, CPU-only PyTorch build, etc.).
    """
    requested = str(cfg.get("device", "cuda")).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[device] Using CUDA: {name} ({mem_gb:.1f} GB)")
            return torch.device("cuda")
        # CUDA requested but not available -- diagnose why
        print("[device] WARNING: CUDA requested but NOT available.")
        print(f"         torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"         torch version = {torch.__version__}")
        print("         Possible causes:")
        print("           - CPU-only PyTorch build (need: pip install torch --index-url https://download.pytorch.org/whl/cu121)")
        print("           - Missing NVIDIA driver or CUDA toolkit")
        print("           - GPU not detected by system")
        print("         Falling back to CPU.")
        return torch.device("cpu")
    print(f"[device] Using CPU (config device='{requested}')")
    return torch.device("cpu")


def ensure_output_dir(cfg: dict) -> str:
    """Create (if needed) and return the output directory for this run.

    Outputs are organised as::

        <output_dir>/<experiment>/<run_id>/

    where ``<experiment>`` comes from ``cfg['experiment']`` (e.g. ``pendulum``,
    ``newton``) and ``<run_id>`` is a timestamped sub-directory that distinguishes
    different runs of the same experiment. This keeps results for every physics
    experiment in its own folder and every run in its own time-stamped folder.

    Resolution of ``run_id`` (in priority order):
      1. ``cfg['run_id']`` if already set (e.g. via ``--run-id`` / ``run_all.py``);
      2. otherwise reuse the most recent run sub-directory for this experiment;
      3. otherwise create a fresh timestamped sub-directory.

    The resolved id is written back into ``cfg['run_id']`` so that every stage
    invoked in the same process (train -> analyze -> symbolic ...) lands in the
    exact same directory.
    """
    base = cfg.get("output_dir", "outputs")
    experiment = str(cfg.get("experiment", "default"))
    run_id = cfg.get("run_id")
    if not run_id:
        exp_dir = os.path.join(base, experiment)
        os.makedirs(exp_dir, exist_ok=True)
        existing = sorted(
            d
            for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d))
        )
        run_id = existing[-1] if existing else make_run_id()
        cfg["run_id"] = run_id  # persist for the rest of this process
    out_dir = os.path.join(base, experiment, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[output] experiment='{experiment}' run='{run_id}' -> {out_dir}")
    return out_dir


def setup_logging(out_dir: str, name: str = "run") -> logging.Logger:
    """Configure a logger that writes to BOTH console and a per-run log file.

    The log file lands at ``<out_dir>/<name>.log`` so every run-id directory
    contains its own complete console transcript for later review.

    Call this ONCE at the start of each stage script (train / analyze / ...),
    after ``ensure_output_dir`` has resolved the output path.

    Returns the configured Logger instance; callers can use it directly or
    just keep using ``print()`` — the root logger's StreamHandler + FileHandler
    will capture everything written via ``logging.info/warning/error`` as well.
    """
    log_path = os.path.join(out_dir, f"{name}.log")

    # Use the root logger so that ALL print-like calls are captured when
    # code uses logging.info() etc. We also add a custom stream wrapper
    # below to capture bare print() statements.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called multiple times (e.g. in tests)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (same as before)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (per-run log)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Redirect bare print() to the logger so existing print-based code
    # is captured without modification.
    class _PrintToLogger:
        """Drop-in replacement for sys.stdout that tees output to a logger."""
        def __init__(self, original_stdout, target_logger):
            self._original = original_stdout
            self._logger = target_logger

        def write(self, msg):
            # Write to real stdout first (so tqdm / interactive output still works)
            self._original.write(msg)
            # Also log non-empty lines (skip pure newlines from tqdm refreshes)
            stripped = msg.rstrip("\n")
            if stripped:
                self._logger.info(stripped)

        def flush(self):
            self._original.flush()

        def __getattr__(self, attr):
            return getattr(self._original, attr)

    sys.stdout = _PrintToLogger(sys.__stdout__, logger)

    logger.info(f"Logging to {log_path}")
    return logger
