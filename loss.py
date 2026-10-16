from __future__ import annotations
import math
import torch
import mmg_model as MM
LAMBDA_POSE_DEFAULT = 0.3

def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def trajectory_loss(z, transform, batch, C, vel_scale, pose_scale, lambda_pose=LAMBDA_POSE_DEFAULT, dt=0.1, eps=MM.SMOOTH_EPS_DEFAULT):
    (s_init, s_tgt, ctrl, wnd) = batch
    params = transform.theta_dict(z)
    pred = MM.rollout(s_init, ctrl, wnd, params, C, dt=dt, eps=eps)
    vel_idx = [1, 3, 5]
    err_vel = (pred[:, :, vel_idx] - s_tgt[:, :, vel_idx]) / vel_scale.view(1, 1, -1)
    loss_vel = torch.mean(err_vel ** 2)
    pdx = pred[:, :, 0] - s_init[:, None, 0]
    pdy = pred[:, :, 2] - s_init[:, None, 2]
    pdpsi = wrap_to_pi(pred[:, :, 4] - s_init[:, None, 4])
    tdx = s_tgt[:, :, 0] - s_init[:, None, 0]
    tdy = s_tgt[:, :, 2] - s_init[:, None, 2]
    tdpsi = wrap_to_pi(s_tgt[:, :, 4] - s_init[:, None, 4])
    pred_pose = torch.stack([pdx, pdy, pdpsi], dim=-1)
    tgt_pose = torch.stack([tdx, tdy, tdpsi], dim=-1)
    err_pose = (pred_pose - tgt_pose) / pose_scale.view(1, 1, -1)
    loss_pose = torch.mean(err_pose ** 2)
    return loss_vel + lambda_pose * loss_pose
if __name__ == '__main__':
    from ship_constants import ShipConstants
    from transform import TanhTransform
    import data as D
    C = ShipConstants()
    Tf = TanhTransform()
    (trajs, _) = D.load_random_trajectories()
    segs = D.make_segments(trajs)
    batch = D.segments_to_tensors(segs)
    (vs, ps) = D.compute_scales(batch[0], batch[1])
    z = Tf.z_efd().clone().requires_grad_(True)
    L = trajectory_loss(z, Tf, batch, C, vs, ps)
    L.backward()
    print(f'EFD-point loss = {L.item():.6f}')
    print(f'grad_z finite = {bool(torch.isfinite(z.grad).all())}, ||grad_z|| = {z.grad.norm().item():.4e}')
