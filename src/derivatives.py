import torch


def grads(y, x):
    """dy/dx with create_graph=True"""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
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
