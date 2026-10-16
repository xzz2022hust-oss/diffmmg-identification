from __future__ import annotations
import os
import pandas as pd
import torch
_CSV_DIR = os.environ.get('DIFFMMG_INPUT_CSV_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_csv'))
_PP_CSV = os.path.join(_CSV_DIR, 'principal_particulars.csv')
_MMG_CSV = os.path.join(_CSV_DIR, 'MMG_params.csv')

def _read_kv_csv(path, key_col, val_col):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    out = {}
    for (_, row) in df.iterrows():
        k = str(row[key_col]).strip()
        out[k] = float(row[val_col])
    return out

class ShipConstants:

    def __init__(self, device='cpu', dtype=torch.float64, pp_csv=_PP_CSV, mmg_csv=_MMG_CSV):
        self.device = device
        self.dtype = dtype
        pp = _read_kv_csv(pp_csv, 'parameter', 'value')
        mp = _read_kv_csv(mmg_csv, 'parameter', 'params_init')
        self.lpp = pp['lpp']
        self.B = pp['breadth']
        self.draft = pp['draft']
        self.mass_nd = pp['mass_nd']
        self.xG_nd = pp['x_lcg']
        self.rho_fresh = pp['rho_fresh'] / 9.80665
        self.dia_prop = pp['dia_prop']
        self.pitchr = pp['pitchr']
        self.area_rudder = pp['area_rudder']
        self.lambda_rudder = pp['lambda_rudder']
        self.x_location_rudder = pp['x_location_rudder']
        self.maxtickness_rudder = pp['maxtickness_rudder']
        self.massx_nd = mp['massx_nd']
        self.massy_nd = mp['massy_nd']
        self.IzzJzz_nd = mp['IzzJzz_nd']
        self.kt_coeff = torch.tensor([mp['kt_coeff0'], mp['kt_coeff1'], mp['kt_coeff2']], device=device, dtype=dtype)
        self.Jmin = mp['Jmin']
        self.alpha_p = mp['alpha_prop']
        self.coeff_urudder_zero = mp['coeff_urudder_zero']
        self.rho_air = pp['rho_air'] / 9.80665
        self.AT = pp['area_projected_trans']
        self.AL = pp['area_projected_lateral']
        self.AOD = pp['AOD']
        self.LCW = pp['LCW']
        self.LCBR = pp['LCBR']
        self.HBR = pp['HBR']
        self.HC = pp['HC']
        self.SBW = pp['SBW']
        self.Lz = pp['Lz']
        self._compute_wind_coefficients()

    def _compute_wind_coefficients(self):
        (lpp, AT, AL) = (self.lpp, self.AT, self.AL)
        (AOD, LCW, LCBR) = (self.AOD, self.LCW, self.LCBR)
        (HBR, HC, SBW) = (self.HBR, self.HC, self.SBW)
        self.XX0 = -0.33 + 0.293 * (SBW * HBR / AT) + 0.0193 * (LCW / HC) + 0.682 * (AOD / lpp ** 2)
        self.XX1 = -1.353 + 1.7 * (AL / lpp / SBW) + 2.87 * (lpp * HC / AL) - 0.463 * (lpp * HBR / AL) - 0.57 * (AOD / AL) - 6.64 * (AT / lpp / SBW) - 0.0123 * (lpp ** 2 / AT) + 0.0202 * (lpp / HC)
        self.XX3 = 0.83 - 0.413 * (AL / lpp / HBR) - 0.0827 * (AL / AT) - 0.563 * (lpp * HC / AL) + 0.804 * (AOD / AL) - 5.67 * (AOD / lpp ** 2) + 0.0401 * (LCW / HC) - 0.132 * (LCBR / lpp)
        self.XX5 = 0.0372 - 0.0075 * (AL / AOD) - 0.103 * (LCBR / lpp) + 0.0921 * (AL / lpp / SBW)
        self.YY1 = 0.684 + 0.717 * (LCBR / lpp) - 3.22 * (LCW / lpp) + 0.0281 * (AL / AOD) + 0.0661 * (LCW / HC) + 0.298 * (AT / (SBW * HBR))
        self.YY3 = -0.4 + 0.282 * (AL / (lpp * SBW)) + 0.307 * (lpp * HC / AL) + 0.0519 * (LCBR / lpp) + 0.0526 * (SBW / HBR) - 0.0814 * (AOD / AL) + 0.0582 * (AT / (SBW * HBR))
        self.YY5 = 0.122 - 0.166 * (AL / (lpp * SBW)) - 0.0054 * (lpp / HBR) - 0.0481 * (LCBR / lpp) - 0.0136 * (SBW ** 2 / AT) + 0.0864 * (LCW / lpp) - 0.0297 * (LCW * HC / AL)
        self.NN1 = 0.299 + 1.71 * (LCW / lpp) + 0.183 * (lpp * HC / AL) - 1.09 * (AT / AL) - 0.0442 * (LCW / HC) - 0.289 * (AL / (lpp * SBW)) + 4.24 * (AT / lpp ** 2) - 0.0646 * (SBW ** 2 / AT) + 0.0306 * (LCBR / lpp)
        self.NN2 = 0.117 + 0.123 * (LCBR / lpp) - 0.323 * (LCW / lpp) + 0.0041 * (AL / AOD) - 0.166 * (AT / SBW ** 2) - 0.0109 * (lpp / HBR) + 0.174 * (AT / (SBW * HBR)) + 0.214 * (AL / (lpp * SBW)) - 1.06 * (AL / lpp ** 2)
        self.NN3 = 0.023 + 0.0385 * (LCBR / lpp) - 0.0339 * (AT / (SBW * HBR)) + 0.0023 * (AL / AT)
if __name__ == '__main__':
    C = ShipConstants()
    print('lpp=%.4f draft=%.5f mass_nd=%.5f xG_nd=%.5f rho_fresh=%.4f' % (C.lpp, C.draft, C.mass_nd, C.xG_nd, C.rho_fresh))
    print('massx=%.4f massy=%.4f IzzJzz=%.4f' % (C.massx_nd, C.massy_nd, C.IzzJzz_nd))
    print('kt_coeff=', C.kt_coeff.tolist())
    print('wind CX0/CX1/CX3/CX5 = %.4f %.4f %.4f %.4f' % (C.XX0, C.XX1, C.XX3, C.XX5))
    print('wind CY1/CY3/CY5 = %.4f %.4f %.4f' % (C.YY1, C.YY3, C.YY5))
    print('wind CN1/CN2/CN3 = %.4f %.4f %.4f' % (C.NN1, C.NN2, C.NN3))
