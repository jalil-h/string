import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -----------------------------
# Config
# -----------------------------
@dataclass
class Params:
    # Dimensionless PDE parameters
    beta: float = 0.3     # V0/c0
    mu: float = 0.02      # eta*L/c0
    Gamma: float = 0.5    # I0*B0*L/P0

    # Time horizon (dimensionless)
    T: float = 6.0

    # Training
    seed: int = 0
    epochs: int = 4000 # 6000
    lr: float = 2e-3

    # Number of sample points per epoch
    N_f: int = 1200   #2000   # PDE collocation points (interior)
    N_b: int = 300    # boundary points (x=0 or x=1 over random t)
    N_i: int = 400    # initial condition points (t=0 over random x)

    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 20.0
    w_ic: float = 20.0
    w_iv: float = 5.0   # initial velocity weight

    # Network
    hidden: int = 64
    depth: int = 5 # 6

    # Output / checkpoint
    out_dir: str = "runs_linear"
    ckpt_name: str = "pinn_linear.pt"
    gif_name: str = "linear_solution.gif"


P = Params()


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_uniform(n: int, low: float, high: float, device):
    return (low + (high - low) * torch.rand(n, 1, device=device))


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden=64, depth=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

        # Xavier init helps smooth PDE training
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Autograd derivatives
# -----------------------------
def grads(y, x):
    """dy/dx with create_graph=True"""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


def compute_derivs(model, x, t):
    """
    Inputs: x,t are [N,1] tensors with requires_grad=True
    Returns dict of required derivatives for wy and wz.
    """
    inp = torch.cat([x, t], dim=1)  # [N,2]
    out = model(inp)               # [N,2]
    wy = out[:, 0:1]
    wz = out[:, 1:2]

    g_wy = grads(wy, inp)
    wy_x = g_wy[:, 0:1]
    wy_t = g_wy[:, 1:2]

    g_wz = grads(wz, inp)
    wz_x = g_wz[:, 0:1]
    wz_t = g_wz[:, 1:2]

    # second derivatives
    wy_xx = grads(wy_x, inp)[:, 0:1]
    wy_tt = grads(wy_t, inp)[:, 1:2]
    wy_xt = grads(wy_x, inp)[:, 1:2]

    wz_xx = grads(wz_x, inp)[:, 0:1]
    wz_tt = grads(wz_t, inp)[:, 1:2]
    wz_xt = grads(wz_x, inp)[:, 1:2]

    return {
        "wy": wy, "wz": wz,
        "wy_x": wy_x, "wy_t": wy_t,
        "wy_xx": wy_xx, "wy_tt": wy_tt, "wy_xt": wy_xt,
        "wz_x": wz_x, "wz_t": wz_t,
        "wz_xx": wz_xx, "wz_tt": wz_tt, "wz_xt": wz_xt,
    }


# -----------------------------
# Linear residuals (dimensionless)
# -----------------------------
def residuals_linear(d, beta, mu, Gamma):
    """
    Linear residuals derived from:
      w_tt + mu w_t + 2 beta w_xt + beta^2 w_xx = w_xx - i Gamma w_x

    Split into wy,wz gives:
      Ry = wy_tt + mu wy_t + 2 beta wy_xt + (beta^2 - 1) wy_xx - Gamma wz_x
      Rz = wz_tt + mu wz_t + 2 beta wz_xt + (beta^2 - 1) wz_xx + Gamma wy_x
    """
    Ry = d["wy_tt"] + mu * d["wy_t"] + 2.0 * beta * d["wy_xt"] + (beta**2 - 1.0) * d["wy_xx"] - Gamma * d["wz_x"]
    Rz = d["wz_tt"] + mu * d["wz_t"] + 2.0 * beta * d["wz_xt"] + (beta**2 - 1.0) * d["wz_xx"] + Gamma * d["wy_x"]
    return Ry, Rz


# -----------------------------
# Initial conditions (choose something simple)
# -----------------------------
def ic_displacement(x):
    """
    w_y(x,0) and w_z(x,0)
    Here: wy0 = sin(pi x), wz0 = 0
    """
    wy0 = torch.sin(math.pi * x)
    wz0 = torch.zeros_like(x)
    return wy0, wz0


def ic_velocity(x):
    """
    w_y_t(x,0) and w_z_t(x,0)
    Here: both zero initial velocities
    """
    vy0 = torch.zeros_like(x)
    vz0 = torch.zeros_like(x)
    return vy0, vz0


# -----------------------------
# Training
# -----------------------------
def train():
    device = torch.device("cpu")
    set_seed(P.seed)

    os.makedirs(P.out_dir, exist_ok=True)

    model = MLP(hidden=P.hidden, depth=P.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=P.lr)

    for ep in range(1, P.epochs + 1):
        opt.zero_grad()

        # 1) PDE collocation points in interior: x in (0,1), t in (0,T)
        x_f = sample_uniform(P.N_f, 0.0, 1.0, device).requires_grad_(True)
        t_f = sample_uniform(P.N_f, 0.0, P.T, device).requires_grad_(True)

        d_f = compute_derivs(model, x_f, t_f)
        Ry, Rz = residuals_linear(d_f, P.beta, P.mu, P.Gamma)
        loss_pde = (Ry**2 + Rz**2).mean()

        # 2) Boundary points: x=0 and x=1, random t
        t_b = sample_uniform(P.N_b, 0.0, P.T, device)
        x0 = torch.zeros_like(t_b).requires_grad_(True)
        x1 = torch.ones_like(t_b).requires_grad_(True)
        t0 = t_b.requires_grad_(True)
        t1 = t_b.requires_grad_(True)

        d0 = compute_derivs(model, x0, t0)
        d1 = compute_derivs(model, x1, t1)
        loss_bc = (d0["wy"]**2 + d0["wz"]**2 + d1["wy"]**2 + d1["wz"]**2).mean()

        # 3) Initial displacement at t=0, random x
        x_i = sample_uniform(P.N_i, 0.0, 1.0, device).requires_grad_(True)
        t_i = torch.zeros_like(x_i).requires_grad_(True)
        d_i = compute_derivs(model, x_i, t_i)
        wy0, wz0 = ic_displacement(x_i)
        loss_ic = ((d_i["wy"] - wy0)**2 + (d_i["wz"] - wz0)**2).mean()

        # 4) Initial velocity at t=0
        vy0, vz0 = ic_velocity(x_i)
        loss_iv = ((d_i["wy_t"] - vy0)**2 + (d_i["wz_t"] - vz0)**2).mean()

        loss = P.w_pde * loss_pde + P.w_bc * loss_bc + P.w_ic * loss_ic + P.w_iv * loss_iv
        loss.backward()
        opt.step()

        if ep % 500 == 0 or ep == 1:
            print(f"ep {ep:5d} | loss={loss.item():.4e} | pde={loss_pde.item():.3e} bc={loss_bc.item():.3e} ic={loss_ic.item():.3e} iv={loss_iv.item():.3e}")

    # Save checkpoint
    ckpt_path = os.path.join(P.out_dir, P.ckpt_name)
    torch.save({"model_state": model.state_dict(), "params": P.__dict__}, ckpt_path)
    print(f"\nSaved: {ckpt_path}")

    return model


# -----------------------------
# Visualization (simple GIF)
# -----------------------------
@torch.no_grad()
def make_gif(model):
    device = torch.device("cpu")

    # evaluation grid
    Nx = 200
    Nt = 160
    x = torch.linspace(0.0, 1.0, Nx, device=device).view(-1, 1)
    times = torch.linspace(0.0, P.T, Nt, device=device)

    # precompute solutions
    wy_all = []
    wz_all = []
    for tval in times:
        t = torch.full_like(x, float(tval))
        out = model(torch.cat([x, t], dim=1))
        wy_all.append(out[:, 0].cpu().numpy())
        wz_all.append(out[:, 1].cpu().numpy())
    wy_all = np.stack(wy_all, axis=0)  # [Nt, Nx]
    wz_all = np.stack(wz_all, axis=0)

    x_np = x[:, 0].cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("w")
    ax.set_title("Linear PINN: wy (solid) and wz (dashed)")

    line_y, = ax.plot(x_np, wy_all[0], lw=2, label="wy")
    line_z, = ax.plot(x_np, wz_all[0], lw=2, ls="--", label="wz")
    ax.legend(loc="upper right")

    def update(k):
        line_y.set_ydata(wy_all[k])
        line_z.set_ydata(wz_all[k])
        ax.set_title(f"Linear PINN: t={times[k].item():.2f}")
        return line_y, line_z

    anim = FuncAnimation(fig, update, frames=Nt, interval=60, blit=True)

    os.makedirs(P.out_dir, exist_ok=True)
    gif_path = os.path.join(P.out_dir, P.gif_name)
    anim.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"Saved GIF: {gif_path}")


@torch.no_grad()
def make_gif_3d(model):
    device = torch.device("cpu")

    Nx = 200
    Nt = 160

    x = torch.linspace(0.0, 1.0, Nx, device=device).view(-1, 1)
    times = torch.linspace(0.0, P.T, Nt, device=device)

    # Precompute solutions
    wy_all = []
    wz_all = []
    for tval in times:
        t = torch.full_like(x, float(tval))
        out = model(torch.cat([x, t], dim=1))
        wy_all.append(out[:, 0].cpu().numpy())
        wz_all.append(out[:, 1].cpu().numpy())
    wy_all = np.stack(wy_all, axis=0)  # [Nt, Nx]
    wz_all = np.stack(wz_all, axis=0)

    x_np = x[:, 0].cpu().numpy()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    # axis limits: keep these reasonable; you can auto-scale later
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("wy")
    ax.set_zlabel("wz")
    ax.set_title("3D string: (x, wy, wz)")

    # initial line
    line, = ax.plot(x_np, wy_all[0], wz_all[0], lw=2)

    def update(k):
        line.set_data(x_np, wy_all[k])
        line.set_3d_properties(wz_all[k])
        ax.set_title(f"3D string: t={times[k].item():.2f}")
        return (line,)

    anim = FuncAnimation(fig, update, frames=Nt, interval=60, blit=True)

    gif_path = os.path.join(P.out_dir, "linear_solution_3d.gif")
    anim.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"Saved 3D GIF: {gif_path}")




def main():
    model = train()
    make_gif(model)
    make_gif_3d(model)


if __name__ == "__main__":
    main()
