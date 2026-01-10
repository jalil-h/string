# Residuals

def residuals_linear(d, beta, mu, Gamma):
    """
    Linear residuals derived from:
      w_tt + mu w_t + 2 beta w_xt + beta^2 w_xx = w_xx - i Gamma w_x

    Split into wy,wz gives:
      Ry = wy_tt + mu wy_t + 2 beta wy_xt + (beta^2 - 1) wy_xx - Gamma wz_x
      Rz = wz_tt + mu wz_t + 2 beta wz_xt + (beta^2 - 1) wz_xx + Gamma wy_x
    """
    Ry = (
        d["wy_tt"]
        + mu * d["wy_t"]
        + 2.0 * beta * d["wy_xt"]
        + (beta**2 - 1.0) * d["wy_xx"]
        - Gamma * d["wz_x"]
    )
    Rz = (
        d["wz_tt"]
        + mu * d["wz_t"]
        + 2.0 * beta * d["wz_xt"]
        + (beta**2 - 1.0) * d["wz_xx"]
        + Gamma * d["wy_x"]
    )
    return Ry, Rz
