# Training implementation
import os
import torch

from .config import P, init_run_dir, save_config
from .utils import set_seed, sample_uniform
from .model import MLP
from .derivatives import compute_derivs
from .residuals import residuals_linear
from .ic import ic_displacement, ic_velocity


def train(device=None):
    if device is None:
        device = torch.device("cpu")

    set_seed(P.seed)

    # Create timestamped run directory once per run
    run_dir = init_run_dir(P)

    # Save config snapshot
    save_config(P, os.path.join(run_dir, "config.json"))

    # Open log file
    log_path = os.path.join(run_dir, "train.log")
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Run directory: {run_dir}")
    log(f"Device: {device}")

    # Initialize model and optimizer
    model = MLP(hidden=P.hidden, depth=P.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=P.lr)

    for ep in range(1, P.epochs + 1):
        opt.zero_grad()

        # 1) PDE collocation points
        x_f = sample_uniform(P.N_f, 0.0, 1.0, device).requires_grad_(True)
        t_f = sample_uniform(P.N_f, 0.0, P.T, device).requires_grad_(True)

        d_f = compute_derivs(model, x_f, t_f)
        Ry, Rz = residuals_linear(d_f, P.beta, P.mu, P.Gamma)
        loss_pde = (Ry**2 + Rz**2).mean()

        # 2) Boundary points
        t_b = sample_uniform(P.N_b, 0.0, P.T, device)
        x0 = torch.zeros_like(t_b).requires_grad_(True)
        x1 = torch.ones_like(t_b).requires_grad_(True)
        t0 = t_b.requires_grad_(True)
        t1 = t_b.requires_grad_(True)

        d0 = compute_derivs(model, x0, t0)
        d1 = compute_derivs(model, x1, t1)
        loss_bc = (d0["wy"]**2 + d0["wz"]**2 + d1["wy"]**2 + d1["wz"]**2).mean()

        # 3) Initial displacement
        x_i = sample_uniform(P.N_i, 0.0, 1.0, device).requires_grad_(True)
        t_i = torch.zeros_like(x_i).requires_grad_(True)
        d_i = compute_derivs(model, x_i, t_i)
        wy0, wz0 = ic_displacement(x_i)
        loss_ic = ((d_i["wy"] - wy0)**2 + (d_i["wz"] - wz0)**2).mean()

        # 4) Initial velocity
        vy0, vz0 = ic_velocity(x_i)
        loss_iv = ((d_i["wy_t"] - vy0)**2 + (d_i["wz_t"] - vz0)**2).mean()

        loss = (
            P.w_pde * loss_pde
            + P.w_bc * loss_bc
            + P.w_ic * loss_ic
            + P.w_iv * loss_iv
        )
        loss.backward()
        opt.step()

        if ep % 500 == 0 or ep == 1:
            log(
                f"ep {ep:5d} | loss={loss.item():.4e} | "
                f"pde={loss_pde.item():.3e} bc={loss_bc.item():.3e} "
                f"ic={loss_ic.item():.3e} iv={loss_iv.item():.3e}"
            )

    # Save checkpoint into run_dir
    ckpt_path = os.path.join(run_dir, P.ckpt_name)
    torch.save({"model_state": model.state_dict(), "params": P.__dict__}, ckpt_path)
    log(f"\nSaved checkpoint: {ckpt_path}")

    log_f.close()
    return model
