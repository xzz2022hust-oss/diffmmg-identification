from __future__ import annotations
import os
import math
import pandas as pd
import torch
_CSV_DIR = os.environ.get('DIFFMMG_INPUT_CSV_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_csv'))
_MMG_CSV = os.path.join(_CSV_DIR, 'MMG_params.csv')
PARAM_NAMES = ['yv_nd', 'yr_nd', 'nv_nd', 'nr_nd', 'xvr_nd', 'coeff_drag_sway', 'cry_cross_flow', 'crn_cross_flow', 'coeff_drag_fore', 'coeff_drag_aft', 't_prop', 'w_prop_zero', 'tau_prop', 'coeff_cp_prop', 'xp_prop_nd', 'ai_coeff_prop0', 'ai_coeff_prop1', 'ai_coeff_prop2', 'ai_coeff_prop3', 'ai_coeff_prop4', 'ai_coeff_prop5', 'ai_coeff_prop6', 'ai_coeff_prop7', 'bi_coeff_prop0', 'bi_coeff_prop1', 'bi_coeff_prop2', 'bi_coeff_prop3', 'bi_coeff_prop4', 'bi_coeff_prop5', 'bi_coeff_prop6', 'bi_coeff_prop7', 'ci_coeff_prop0', 'ci_coeff_prop1', 'ci_coeff_prop2', 'ci_coeff_prop3', 't_rudder', 'ah_rudder', 'xh_rudder_nd', 'lr_rudder_nd', 'kx_rudder', 'kx_rudder_reverse', 'epsilon_rudder', 'cpr_rudder', 'gammaN_rudder', 'gammaP_rudder']
assert len(PARAM_NAMES) == 45, f'expected 45, got {len(PARAM_NAMES)}'
assert len(set(PARAM_NAMES)) == 45, 'duplicate param name'
_SIGN_NEG = {'yv_nd', 'nr_nd'}
_SIGN_POS = {'t_prop', 'w_prop_zero'}
APPLY_GEOM_SIGN_PRIOR = True
_SIGN_NEG_GEOM = {'xp_prop_nd', 'xh_rudder_nd', 'coeff_drag_fore', 'coeff_drag_aft'}
_NEAR_ZERO = 1e-06

def load_efd():
    df = pd.read_csv(_MMG_CSV)
    df.columns = df.columns.str.strip()
    df['parameter'] = df['parameter'].astype(str).str.strip()
    m = dict(zip(df['parameter'], df['params_init'].astype(float)))
    return {k: float(m[k]) for k in PARAM_NAMES}

def build_bounds(efd: dict):
    (lo, hi) = ([], [])
    for name in PARAM_NAMES:
        e = efd[name]
        if abs(e) < _NEAR_ZERO:
            raise ValueError(f'{name} EFD={e} near-zero -> multiplicative bound collapses; give an absolute physical bound.')
        neg_lock = name in _SIGN_NEG or (APPLY_GEOM_SIGN_PRIOR and name in _SIGN_NEG_GEOM)
        if neg_lock:
            if e >= 0:
                raise ValueError(f'{name} sign-neg prior but EFD={e}>=0')
            (l, h) = (10.0 * e, 0.1 * e)
        elif name in _SIGN_POS:
            (l, h) = (0.1 * e, 10.0 * e)
        else:
            (l, h) = (-10.0 * abs(e), 10.0 * abs(e))
        if l > h:
            (l, h) = (h, l)
        lo.append(l)
        hi.append(h)
    return (lo, hi)

class TanhTransform:

    def __init__(self, device='cpu', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.efd = load_efd()
        (lo, hi) = build_bounds(self.efd)
        self.lo = torch.tensor(lo, device=device, dtype=dtype)
        self.hi = torch.tensor(hi, device=device, dtype=dtype)
        self.m = 0.5 * (self.lo + self.hi)
        self.h = 0.5 * (self.hi - self.lo)
        self.theta_efd = torch.tensor([self.efd[n] for n in PARAM_NAMES], device=device, dtype=dtype)

    def to_theta(self, z: torch.Tensor) -> torch.Tensor:
        return self.m + self.h * torch.tanh(z)

    def to_z(self, theta: torch.Tensor) -> torch.Tensor:
        s = (theta - self.m) / self.h
        s = torch.clamp(s, min=-1 + 1e-09, max=1 - 1e-09)
        return torch.atanh(s)

    def z_efd(self) -> torch.Tensor:
        return self.to_z(self.theta_efd)

    def theta_dict(self, z: torch.Tensor) -> dict:
        theta = self.to_theta(z)
        if theta.dim() == 1:
            return {n: theta[i] for (i, n) in enumerate(PARAM_NAMES)}
        return {n: theta[..., i] for (i, n) in enumerate(PARAM_NAMES)}
if __name__ == '__main__':
    T = TanhTransform()
    efd = T.theta_efd
    z0 = T.z_efd()
    theta_rt = T.to_theta(z0)
    max_err = (theta_rt - efd).abs().max().item()
    print(f'n_params = {len(PARAM_NAMES)}')
    print(f'roundtrip max|θ_EFD - T(T^-1(θ_EFD))| = {max_err:.3e}')
    print(f'min |EFD| = {efd.abs().min().item():.3e}  (near-zero thr {_NEAR_ZERO})')
    s = (efd - T.m) / T.h
    keep = 1 - s ** 2
    worst = keep.argmin().item()
    print(f'worst-saturation param: {PARAM_NAMES[worst]}  s={s[worst]:.3f}  1-s^2={keep[worst]:.3f}')
    print(f'median 1-s^2 across 45 params = {keep.median().item():.3f}')
