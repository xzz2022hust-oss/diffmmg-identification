from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import torch
DATA_ROOT = os.environ.get('DIFFMMG_DATA_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
DT = 0.1
SEG_LEN = 50

def load_random_trajectories(data_root=DATA_ROOT):
    files = sorted(glob.glob(os.path.join(data_root, 'random', '*.csv')))
    trajs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df = df.dropna().reset_index(drop=True)
        st = np.column_stack([df['x_position_mid [m]'].values, df['u_velo [m/s]'].values, df['y_position_mid [m]'].values, df['vm_velo [m/s]'].values, df['psi_hat [rad]'].values, df['r_angvelo [rad/s]'].values])
        ctrl = np.column_stack([df['delta_rudder [rad]'].values, df['n_prop [rps]'].values])
        wnd = np.column_stack([df['wind_velo_relative_mid [m/s]'].values, df['wind_dir_relative_mid [rad]'].values])
        trajs.append({'state': st, 'control': ctrl, 'wind': wnd, 'name': os.path.basename(f)})
    return (trajs, files)

def make_segments(trajs, seg_len=SEG_LEN, stride=None):
    if stride is None:
        stride = seg_len
    segs = []
    for tr in trajs:
        Tn = len(tr['state'])
        for t0 in range(0, Tn - seg_len, stride):
            t1 = t0 + seg_len + 1
            if t1 > Tn:
                break
            segs.append({'state': tr['state'][t0:t1], 'control': tr['control'][t0:t0 + seg_len], 'wind': tr['wind'][t0:t0 + seg_len]})
    return segs

def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def segments_to_tensors(segs, device='cpu', dtype=torch.float64):
    s_init = np.stack([s['state'][0] for s in segs])
    s_tgt = np.stack([s['state'][1:] for s in segs])
    ctrl = np.stack([s['control'] for s in segs])
    wnd = np.stack([s['wind'] for s in segs])
    return (torch.tensor(s_init, device=device, dtype=dtype), torch.tensor(s_tgt, device=device, dtype=dtype), torch.tensor(ctrl, device=device, dtype=dtype), torch.tensor(wnd, device=device, dtype=dtype))

def compute_scales(s_init, s_tgt, device='cpu', dtype=torch.float64):
    vel_idx = [1, 3, 5]
    vel_scale = s_tgt[:, :, vel_idx].reshape(-1, 3).std(dim=0, unbiased=False).clamp_min(1e-06)
    dx = s_tgt[:, :, 0] - s_init[:, None, 0]
    dy = s_tgt[:, :, 2] - s_init[:, None, 2]
    dpsi = (s_tgt[:, :, 4] - s_init[:, None, 4] + np.pi) % (2 * np.pi) - np.pi
    pose = torch.stack([dx, dy, dpsi], dim=-1).reshape(-1, 3)
    pose_scale = pose.std(dim=0, unbiased=False).clamp_min(1e-06)
    return (vel_scale.to(device, dtype), pose_scale.to(device, dtype))
if __name__ == '__main__':
    (trajs, files) = load_random_trajectories()
    segs = make_segments(trajs)
    print(f'n_random_trials = {len(trajs)}')
    print(f"total_rows = {sum((len(t['state']) for t in trajs))}")
    print(f'n_random_segments = {len(segs)} (seg_len={SEG_LEN}, non-overlap)')
    (si, st, ct, wd) = segments_to_tensors(segs)
    (vs, ps) = compute_scales(si, st)
    print('state_init:', tuple(si.shape), ' state_tgt:', tuple(st.shape))
    print('vel_scale =', vs.tolist())
    print('pose_scale =', ps.tolist())
