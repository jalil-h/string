# Initial conditions

import math
import torch


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
