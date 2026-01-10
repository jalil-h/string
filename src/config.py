# Configuration file
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os


def make_run_dir(base_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(base_dir, ts)


@dataclass
class Params:
    # Dimensionless PDE parameters
    beta: float = 0.3
    mu: float = 0.02
    Gamma: float = 0.5

    # Time horizon (dimensionless)
    T: float = 6.0

    # Training
    seed: int = 0
    epochs: int = 4000
    lr: float = 2e-3

    # Number of sample points per epoch
    N_f: int = 1200
    N_b: int = 300
    N_i: int = 400

    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 20.0
    w_ic: float = 20.0
    w_iv: float = 5.0

    # Network
    hidden: int = 64
    depth: int = 5

    # Output base directory
    out_dir: str = "runs_linear"
    ckpt_name: str = "pinn_linear.pt"
    gif_name: str = "linear_solution.gif"
    gif3d_name: str = "linear_solution_3d.gif"

    # Runtime-created per run
    run_dir: str | None = None  # filled in at runtime


P = Params()


def init_run_dir(P: Params) -> str:
    """Create and store a timestamped run directory."""
    run_dir = make_run_dir(P.out_dir)
    os.makedirs(run_dir, exist_ok=True)
    P.run_dir = run_dir
    return run_dir


def save_config(P: Params, path: str):
    """Save Params to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(P), f, indent=2)
