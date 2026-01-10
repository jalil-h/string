# Visualization

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .config import P


@torch.no_grad()
def make_gif(model, device=None):
    if device is None:
        device = torch.device("cpu")
    if not P.run_dir:
        raise RuntimeError("P.run_dir is not set. Run train() first.")

    Nx = 200
    Nt = 160
    x = torch.linspace(0.0, 1.0, Nx, device=device).view(-1, 1)
    times = torch.linspace(0.0, P.T, Nt, device=device)

    wy_all, wz_all = [], []
    for tval in times:
        t = torch.full_like(x, float(tval))
        out = model(torch.cat([x, t], dim=1))
        wy_all.append(out[:, 0].cpu().numpy())
        wz_all.append(out[:, 1].cpu().numpy())
    wy_all = np.stack(wy_all, axis=0)
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

    gif_path = os.path.join(P.run_dir, P.gif_name)
    anim.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"Saved GIF: {gif_path}")


@torch.no_grad()
def make_gif_3d(model, device=None):
    if device is None:
        device = torch.device("cpu")
    if not P.run_dir:
        raise RuntimeError("P.run_dir is not set. Run train() first.")

    Nx = 200
    Nt = 160
    x = torch.linspace(0.0, 1.0, Nx, device=device).view(-1, 1)
    times = torch.linspace(0.0, P.T, Nt, device=device)

    wy_all, wz_all = [], []
    for tval in times:
        t = torch.full_like(x, float(tval))
        out = model(torch.cat([x, t], dim=1))
        wy_all.append(out[:, 0].cpu().numpy())
        wz_all.append(out[:, 1].cpu().numpy())
    wy_all = np.stack(wy_all, axis=0)
    wz_all = np.stack(wz_all, axis=0)

    x_np = x[:, 0].cpu().numpy()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("wy")
    ax.set_zlabel("wz")
    ax.set_title("3D string: (x, wy, wz)")

    line, = ax.plot(x_np, wy_all[0], wz_all[0], lw=2)

    def update(k):
        line.set_data(x_np, wy_all[k])
        line.set_3d_properties(wz_all[k])
        ax.set_title(f"3D string: t={times[k].item():.2f}")
        return (line,)

    anim = FuncAnimation(fig, update, frames=Nt, interval=60, blit=True)

    gif_path = os.path.join(P.run_dir, P.gif3d_name)
    anim.save(gif_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"Saved 3D GIF: {gif_path}")
