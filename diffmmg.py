from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
from ship_constants import ShipConstants
from transform import TanhTransform, PARAM_NAMES
import data as D
import loss as LOSS

def set_threads(p):
    torch.set_num_threads(int(p))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    os.environ['OMP_NUM_THREADS'] = str(p)
    os.environ['MKL_NUM_THREADS'] = str(p)

def run(n_steps=500, lr=0.001, seed=0, threads=None, grad_clip=10.0, lambda_pose=0.3, log_every=25, save_dir=None, z0=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if threads:
        set_threads(threads)
    C = ShipConstants()
    Tf = TanhTransform()
    (trajs, _) = D.load_random_trajectories()
    if not trajs:
        raise FileNotFoundError(f'No random-steering CSV files found under {D.DATA_ROOT!r}.')
    segs = D.make_segments(trajs)
    batch = D.segments_to_tensors(segs)
    (vs, ps) = D.compute_scales(batch[0], batch[1])
    z = (Tf.z_efd().clone() if z0 is None else z0.clone()).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    history = []
    t0 = time.perf_counter()
    best = float('inf')
    best_z = z.detach().clone()
    for step in range(n_steps):
        opt.zero_grad()
        L = LOSS.trajectory_loss(z, Tf, batch, C, vs, ps, lambda_pose=lambda_pose)
        lv = L.item()
        if lv < best:
            best = lv
            best_z = z.detach().clone()
        L.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
        opt.step()
        el = time.perf_counter() - t0
        history.append({'n_grad': step + 1, 'loss': lv, 'wall_s': el})
        if step % log_every == 0 or step == n_steps - 1:
            print(f'[Diff step {step:4d}] loss={lv:.6f}  best={best:.6f}  {el:.1f}s', flush=True)
    with torch.no_grad():
        final_loss = LOSS.trajectory_loss(z, Tf, batch, C, vs, ps, lambda_pose=lambda_pose).item()
    if final_loss < best:
        best = final_loss
        best_z = z.detach().clone()
    total = time.perf_counter() - t0
    print(f'DiffMMG done: N_grad={n_steps}, best_loss={best:.6f}, wall={total:.1f}s')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        theta_best = Tf.to_theta(best_z).detach().numpy().tolist()
        json.dump({'optimizer': 'adam_on_z', 'seed': seed, 'n_grad': n_steps, 'lr': lr, 'grad_clip': grad_clip, 'lambda_pose': lambda_pose, 'threads': threads, 'best_loss': best, 'final_loss': final_loss, 'wall_s': total, 'param_names': PARAM_NAMES, 'theta_best': theta_best, 'history': history}, open(os.path.join(save_dir, f'diffmmg_seed{seed}.json'), 'w'), indent=2)
        print('saved:', os.path.join(save_dir, f'diffmmg_seed{seed}.json'))
    return (best, history)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_steps', type=int, default=500)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--save_dir', type=str, default=None)
    a = ap.parse_args()
    run(n_steps=a.n_steps, lr=a.lr, seed=a.seed, threads=a.threads, save_dir=a.save_dir)
