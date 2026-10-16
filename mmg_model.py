from __future__ import annotations
import math
import torch
SMOOTH_EPS_DEFAULT = 1e-07

def smooth_abs(x, eps=SMOOTH_EPS_DEFAULT):
    return torch.sqrt(x * x + eps)

def _g(params, name):
    return params[name]

def hull_force(u, vm, r, beta, params, C, eps):
    (lpp, draft, rho) = (C.lpp, C.draft, C.rho_fresh)
    U_safe = torch.clamp(torch.sqrt(u * u + vm * vm), min=1e-05)
    x_int = torch.linspace(-lpp / 2, lpp / 2, 100, device=u.device, dtype=u.dtype)
    CrY = _g(params, 'cry_cross_flow')
    CrN = _g(params, 'crn_cross_flow')
    CD = _g(params, 'coeff_drag_sway')
    CrY = CrY.unsqueeze(-1) if torch.is_tensor(CrY) and CrY.dim() > 0 else CrY
    CrN = CrN.unsqueeze(-1) if torch.is_tensor(CrN) and CrN.dim() > 0 else CrN
    comp_y = vm.unsqueeze(-1) + CrY * r.unsqueeze(-1) * x_int
    comp_n = vm.unsqueeze(-1) + CrN * r.unsqueeze(-1) * x_int
    int_y = smooth_abs(comp_y, eps) * comp_y
    int_n = smooth_abs(comp_n, eps) * comp_n * x_int
    y_cross = torch.trapezoid(int_y, x_int, dim=-1) / lpp * CD
    n_cross = torch.trapezoid(int_n, x_int, dim=-1) / lpp ** 2 * CD
    dim_uv = 0.5 * rho * lpp * draft
    dim_r = 0.5 * rho * lpp ** 2 * draft
    X0F = _g(params, 'coeff_drag_fore')
    X0A = _g(params, 'coeff_drag_aft')
    Xvr = _g(params, 'xvr_nd')
    Yv = _g(params, 'yv_nd')
    Yr = _g(params, 'yr_nd')
    Nv = _g(params, 'nv_nd')
    Nr = _g(params, 'nr_nd')
    XH = dim_uv * ((X0F + (X0A - X0F) * smooth_abs(beta, eps) / math.pi) * u * U_safe + Xvr * vm * r * lpp)
    YH = dim_uv * (Yv * vm * smooth_abs(u, eps) + Yr * r * lpp * u - y_cross)
    NH = dim_r * (Nv * vm * u + Nr * r * lpp * smooth_abs(u, eps) - n_cross)
    return (XH, YH, NH)

def propeller_force(u, vm, r, n_prop, params, C, eps):
    (lpp, draft, rho) = (C.lpp, C.draft, C.rho_fresh)
    dp = C.dia_prop
    pitch = dp * C.pitchr
    kt = C.kt_coeff
    U_safe = torch.clamp(torch.sqrt(u * u + vm * vm), min=1e-05)
    v_nd = vm / U_safe
    r_nd = r * lpp / U_safe
    wP0 = _g(params, 'w_prop_zero')
    tau = _g(params, 'tau_prop')
    CP = _g(params, 'coeff_cp_prop')
    xP = _g(params, 'xp_prop_nd')
    wP = wP0 - tau * smooth_abs(v_nd + xP * r_nd, eps) - CP * (v_nd + xP * r_nd) ** 2
    one_minus_wp = torch.where(u > 0, 1.0 - wP, torch.ones_like(wP))
    one_minus_wp = torch.clamp(one_minus_wp, min=1e-07, max=1.0)
    n_safe = torch.where(torch.abs(n_prop) < 1e-10, torch.full_like(n_prop, 1e-10), n_prop)
    Js = u / (dp * n_safe)
    J_prop = torch.where(u >= 0, Js * one_minus_wp, Js)
    t_prop = _g(params, 't_prop')
    KT_1st = kt[0] + kt[1] * J_prop + kt[2] * J_prop ** 2
    Ci = [_g(params, f'ci_coeff_prop{i}') for i in range(4)]
    XP_fwd = rho * dp ** 4 * n_prop ** 2 * (1 - t_prop) * KT_1st
    XP_rev = torch.where(Js >= Ci[3], rho * dp ** 4 * n_prop ** 2 * (Ci[1] + Ci[2] * Js), rho * dp ** 4 * n_prop ** 2 * Ci[0])
    XP = torch.where(n_prop >= 0, XP_fwd, XP_rev)
    Ai = [_g(params, f'ai_coeff_prop{i}') for i in range(8)]
    Bi = [_g(params, f'bi_coeff_prop{i}') for i in range(8)]
    (Jsyn, Jsyn0) = (-0.35, -0.06)
    dim_y = 0.5 * rho * lpp * draft
    dim_n = 0.5 * rho * lpp ** 2 * draft
    YP_2nd = dim_y * (n_prop * pitch) ** 2 * (Ai[5] * Js ** 2 + Ai[6] * Js + Ai[7])
    YP_n_pos = torch.where(u >= 0, torch.zeros_like(u), YP_2nd)
    YP_n_neg = torch.where(Js < Jsyn, dim_y * (n_prop * dp) ** 2 * (Ai[2] + Ai[3] * Js), torch.where(Js > Jsyn0, dim_y * (n_prop * dp) ** 2 * Ai[4], dim_y * (n_prop * dp) ** 2 * (Ai[0] + Ai[1] * Js)))
    YP = torch.where(n_prop >= 0, YP_n_pos, YP_n_neg)
    NP_2nd = dim_n * (n_prop * pitch) ** 2 * (Bi[5] * Js ** 2 + Bi[6] * Js + Bi[7])
    NP_n_pos = torch.where(u >= 0, torch.zeros_like(u), NP_2nd)
    NP_n_neg = torch.where(Js < Jsyn, dim_n * (n_prop * dp) ** 2 * (Bi[2] + Bi[3] * Js), torch.where(Js > Jsyn0, dim_n * (n_prop * dp) ** 2 * Bi[4], dim_n * (n_prop * dp) ** 2 * (Bi[0] + Bi[1] * Js)))
    NP = torch.where(n_prop >= 0, NP_n_pos, NP_n_neg)
    return (XP, YP, NP, one_minus_wp, J_prop)

def _rudder_inflow(u, vm, r, delta, n_prop, one_minus_wp, XP, params, C, eps):
    (lpp, dp, rho) = (C.lpp, C.dia_prop, C.rho_fresh)
    t_prop = _g(params, 't_prop')
    area_R = C.area_rudder
    h_R = math.sqrt(area_R * C.lambda_rudder)
    eta_R = dp / h_R
    lR = _g(params, 'lr_rudder_nd') * lpp
    uprop = u * one_minus_wp
    eps_r = _g(params, 'epsilon_rudder')
    kx = _g(params, 'kx_rudder')
    kappa = kx / eps_r
    one_minus_wR = eps_r * one_minus_wp
    n_sq_safe = torch.clamp(n_prop ** 2, min=1e-14)
    KT_pos = XP / (rho * dp ** 4 * n_sq_safe * (1 - t_prop))
    KT_neg = XP / (rho * dp ** 4 * n_sq_safe)
    KT = torch.where(n_prop > 1e-10, KT_pos, torch.where(torch.abs(n_prop) > 1e-10, KT_neg, torch.zeros_like(u)))
    KT = torch.clamp(KT, min=-5.0, max=5.0)
    inner = uprop ** 2 + 8 * KT * n_prop ** 2 * dp ** 2 / math.pi
    inner_safe = torch.clamp(inner, min=0.0)
    uR_1st = eps_r * torch.sqrt(eta_R * (uprop + kappa * (torch.sqrt(inner_safe) - uprop)) ** 2 + (1 - eta_R) * uprop ** 2)
    kxr = _g(params, 'kx_rudder_reverse')
    cpr = _g(params, 'cpr_rudder')
    urpr1 = u * one_minus_wR + n_prop * dp * kxr * torch.sqrt(8 * smooth_abs(KT, eps) / math.pi)
    urpr2 = u * one_minus_wR
    ursq = eta_R * torch.sign(urpr1) * urpr1 ** 2 + (1 - eta_R) * urpr2 ** 2 + cpr * u ** 2
    uR_rev_fwd = torch.sign(ursq) * torch.sqrt(smooth_abs(ursq, eps))
    uR_rev = torch.where(u >= 0, uR_rev_fwd, u)
    uR = torch.where((n_prop >= 0) & (KT > 0), uR_1st, uR_rev)
    x_rud = C.x_location_rudder
    cond_v = vm + x_rud * r
    gammaP = _g(params, 'gammaP_rudder')
    gammaN = _g(params, 'gammaN_rudder')
    vR = torch.where(cond_v >= 0, -gammaP * (vm + lR * r), -gammaN * (vm + lR * r))
    U_R = torch.sqrt(uR ** 2 + vR ** 2)
    aR = delta - torch.atan2(vR, uR)
    return (aR, U_R)

def rudder_force(u, vm, r, delta, n_prop, one_minus_wp, XP, params, C, eps):
    (lpp, rho) = (C.lpp, C.rho_fresh)
    area_R = C.area_rudder
    lam_R = C.lambda_rudder
    x_rud = C.x_location_rudder
    (aR, U_R) = _rudder_inflow(u, vm, r, delta, n_prop, one_minus_wp, XP, params, C, eps)
    cn = 6.13 * lam_R / (2.25 + lam_R) * torch.sin(aR)
    FN = 0.5 * rho * area_R * U_R ** 2 * cn
    t_rud = _g(params, 't_rudder')
    aH = _g(params, 'ah_rudder')
    xH = _g(params, 'xh_rudder_nd') * lpp
    XR = -(1 - t_rud) * FN * torch.sin(delta)
    YR = -(1 + aH) * FN * torch.cos(delta)
    NR = -(x_rud + aH * xH) * FN * torch.cos(delta)
    return (XR, YR, NR)

def wind_force(U_A, gamma_A, C):
    aoa = 2.0 * math.pi - gamma_A
    CX = C.XX0 + C.XX1 * torch.cos(aoa) + C.XX3 * torch.cos(3 * aoa) + C.XX5 * torch.cos(5 * aoa)
    CY = C.YY1 * torch.sin(aoa) + C.YY3 * torch.sin(3 * aoa) + C.YY5 * torch.sin(5 * aoa)
    CN = C.NN1 * torch.sin(aoa) + C.NN2 * torch.sin(2 * aoa) + C.NN3 * torch.sin(3 * aoa)
    q = 0.5 * C.rho_air * U_A ** 2
    XA = CX * q * C.AT
    YA = CY * q * C.AL
    NA = CN * q * C.lpp * C.AL
    return (XA, YA, NA)

def mmg_accel(u, vm, r, psi, delta, n_prop, U_A, gamma_A, params, C, eps=SMOOTH_EPS_DEFAULT):
    (lpp, draft, rho) = (C.lpp, C.draft, C.rho_fresh)
    beta = torch.atan2(vm, u)
    (XH, YH, NH) = hull_force(u, vm, r, beta, params, C, eps)
    (XP, YP, NP, omwp, Jp) = propeller_force(u, vm, r, n_prop, params, C, eps)
    (XR, YR, NR) = rudder_force(u, vm, r, delta, n_prop, omwp, XP, params, C, eps)
    (XA, YA, NA) = wind_force(U_A, gamma_A, C)
    X = XH + XP + XR + XA
    Y = YH + YP + YR + YA
    N = NH + NP + NR + NA
    dim_M = 0.5 * rho * lpp ** 2 * draft
    dim_I = 0.5 * rho * lpp ** 4 * draft
    Mass = C.mass_nd * dim_M
    MassX = C.massx_nd * dim_M
    MassY = C.massy_nd * dim_M
    IJzz = C.IzzJzz_nd * dim_I
    xG = C.xG_nd * lpp
    u_dot = (X + (Mass + MassY) * vm * r + xG * Mass * r ** 2) / (Mass + MassX)
    AA1 = Mass + MassY
    AA2 = xG * Mass
    AA3 = Y - (Mass + MassX) * u * r
    BB1 = IJzz + xG ** 2 * Mass
    BB3 = N - xG * Mass * u * r
    denom = AA1 * BB1 - AA2 * AA2
    vm_dot = (AA3 * BB1 - AA2 * BB3) / denom
    r_dot = (AA3 * AA2 - BB3 * AA1) / (AA2 * AA2 - AA1 * BB1)
    x_dot = u * torch.cos(psi) - vm * torch.sin(psi)
    y_dot = u * torch.sin(psi) + vm * torch.cos(psi)
    return (u_dot, vm_dot, r_dot, x_dot, y_dot)

def rk4_step(state, delta, n_prop, U_A, gamma_A, params, C, dt=0.1, eps=SMOOTH_EPS_DEFAULT):

    def deriv(s):
        (x, u, y, vm, psi, r) = (s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5])
        (ud, vd, rd, xd, yd) = mmg_accel(u, vm, r, psi, delta, n_prop, U_A, gamma_A, params, C, eps)
        return torch.stack([xd, ud, yd, vd, r, rd], dim=-1)
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

def rollout(state_init, control, wind, params, C, dt=0.1, eps=SMOOTH_EPS_DEFAULT):
    T = control.shape[1]
    preds = []
    s = state_init
    for t in range(T):
        s = rk4_step(s, control[:, t, 0], control[:, t, 1], wind[:, t, 0], wind[:, t, 1], params, C, dt=dt, eps=eps)
        preds.append(s)
    return torch.stack(preds, dim=1)
if __name__ == '__main__':
    from ship_constants import ShipConstants
    from transform import TanhTransform
    C = ShipConstants()
    T = TanhTransform()
    params = T.theta_dict(T.z_efd())
    N = 4
    s0 = torch.zeros(N, 6, dtype=torch.float64)
    s0[:, 1] = 0.3
    ctrl = torch.zeros(N, 5, 2, dtype=torch.float64)
    ctrl[:, :, 1] = 8.0
    wnd = torch.zeros(N, 5, 2, dtype=torch.float64)
    out = rollout(s0, ctrl, wnd, params, C)
    print('rollout out shape:', tuple(out.shape))
    print('finite:', bool(torch.isfinite(out).all()))
    print('u after 5 steps:', out[0, :, 1].tolist())
