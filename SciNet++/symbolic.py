"""
symbolic.py
===========
SciNet++ (2026) -- Stage 3: AI-Feynman-style formula discovery with PySR.

Takes the frozen latent concepts saved by ``analyze.py`` and runs symbolic
regression (PySR) to search for a closed-form expression mapping the learned
latent vector z to each ground-truth physical concept. This is the modern
"AI Feynman" step: instead of hoping a human interprets the latent axes, we let
a symbolic engine propose an explicit equation.

PySR is a Julia-backed package; on first use it installs its Julia toolchain
automatically (this can take a few minutes). If PySR is unavailable the script
degrades gracefully with installation guidance instead of crashing.

Inputs (produced by analyze.py, in cfg['output_dir']):
    latent.npy, labels.npy, label_names.json

Outputs:
    equations_<concept>.txt   human-readable PySR result table
    equations_<concept>.csv   full candidate front (model.equations_)

Usage:
    python symbolic.py --config configs/pendulum.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from utils.config import ensure_output_dir, load_config, make_run_id, setup_logging

try:  # PySR pulls in a Julia backend; keep the import optional.
    from pysr import PySRRegressor

    PYSR_AVAILABLE = True
    PYSR_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    PYSR_AVAILABLE = False
    PYSR_ERROR = exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbolic regression for SciNet++")
    parser.add_argument("--config", default="configs/pendulum.yaml")
    parser.add_argument(
        "--run-id",
        default=None,
        help="reuse a specific run sub-directory (timestamp); auto-generated if omitted",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="force a fresh timestamped run sub-directory even if one exists",
    )
    return parser.parse_args()


def load_concepts(out_dir: str):
    """Load frozen latents and ground-truth concept vectors saved by analyze.py.

    The data flow is:
      train.py -> checkpoint.pt
      analyze.py -> latent.npy (N, latent_dim) + labels.npy (N, num_concepts)
      symbolic.py reads these files and runs PySR: latent -> concept formula

    This is the AI Feynman handoff: we don't need to know what the latent dims
    mean -- PySR searches for ANY mathematical expression that maps z to the
    ground-truth concept value.
    """
    latent_path = os.path.join(out_dir, "latent.npy")
    labels_path = os.path.join(out_dir, "labels.npy")
    if not (os.path.exists(latent_path) and os.path.exists(labels_path)):
        raise FileNotFoundError(
            f"Missing {latent_path} / {labels_path}. Run analyze.py first."
        )
    Z = np.load(latent_path)
    Y = np.load(labels_path)

    names_path = os.path.join(out_dir, "label_names.json")
    if os.path.exists(names_path):
        with open(names_path, "r", encoding="utf-8") as fh:
            names = json.load(fh)
    else:
        names = [f"concept_{j}" for j in range(Y.shape[1])]
    return Z, Y, names


def discover_formula(Z: np.ndarray, y: np.ndarray, cfg: dict,
                     out_dir: str = ".") -> "PySRRegressor":
    """Fit a PySR symbolic regression from latents to one concept.

    PySR uses genetic programming to search over mathematical expressions,
    optimising a Pareto front of (accuracy, simplicity). It tries combinations
    of the configured operators (+, -, *, exp, cos, etc.) applied to the
    latent dimensions x1, x2, ..., and returns equations ranked by a score
    that balances fit quality against complexity.

    IMPORTANT: The input features are the LATENT dims (z0, z1, ...), NOT the
    raw time-series. So the discovered formula maps *learned representations*
    to physical concepts, e.g.:
        gamma ≈ f(z0, z1, z2, z3)
    If R^2 is high in analyze.py, PySR should find a simple, accurate formula.

    Parameters
    ----------
    out_dir : str
        Directory where PySR should store its temporary hall_of_fame files.
        Without this, PySR creates randomly-named directories in the working
        directory (e.g. ``outputs/20260916_..._fyONNI/``), which pollutes the
        output tree and makes it impossible to tell which experiment they belong to.
    """
    s = cfg["symbolic"]
    model = PySRRegressor(
        niterations=s.get("niterations", 40),       # generations of evolution
        binary_operators=s.get("binary_operators", ["+", "-", "*"]),  # allowed ops
        unary_operators=s.get("unary_operators", ["exp", "cos"]),     # allowed funcs
        maxsize=s.get("max_complexity", 20),         # max expression tree size
        batch_size=s.get("batch_size", 50),           # subsample per iteration
        random_state=cfg.get("seed", 42),
        verbosity=1,
        progress=False,
        # Keep PySR's temp files inside our managed output directory so they
        # don't create orphan directories at the project root. delete_tempfiles
        # ensures they are cleaned up after the search completes.
        temp_equation_file=True,
        delete_tempfiles=True,
    )
    model.fit(Z, y)  # Z = (N, latent_dim), y = (N,) concept values
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.new:
        cfg["run_id"] = make_run_id()
    elif args.run_id:
        cfg["run_id"] = args.run_id
    out_dir = ensure_output_dir(cfg)
    setup_logging(out_dir, name="symbolic")

    if not PYSR_AVAILABLE:
        print("=" * 70)
        print("PySR is not available in this environment.")
        print("Install with:  pip install pysr")
        print("then run once:  python -c \"import pysr; pysr.PySRRegressor()\"")
        print("(PySR downloads a self-contained Julia backend on first use.)")
        print(f"Import error was: {PYSR_ERROR!r}")
        print("=" * 70)
        sys.exit(0)

    Z, Y, names = load_concepts(out_dir)
    print(f"Loaded latents {Z.shape} and {len(names)} concept target(s): {names}")

    for j, name in enumerate(names):
        print(f"\n=== Symbolic regression: latent -> {name} ===")
        # Run PySR: search for formula mapping latent dims -> this concept
        model = discover_formula(Z, Y[:, j], cfg, out_dir=out_dir)

        # Best equation = highest score on the Pareto front (best accuracy/simplicity tradeoff)
        best = model.get_best()
        print(f"Best equation for {name}:\n{best}")

        txt_path = os.path.join(out_dir, f"equations_{name}.txt")
        csv_path = os.path.join(out_dir, f"equations_{name}.csv")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"# PySR result for concept: {name}\n")
            fh.write(f"# features: latent dims z0..z{Z.shape[1] - 1}\n\n")
            fh.write(str(model))
        try:
            model.equations_.to_csv(csv_path)
        except Exception:  # pragma: no cover - pandas/attribute differences
            pass
        print(f"Saved {txt_path}")


if __name__ == "__main__":
    main()
