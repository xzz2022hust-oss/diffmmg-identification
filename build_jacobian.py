from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.func as torch_func

import data as dataset
import mmg_model as mmg
from ship_constants import ShipConstants
from transform import PARAM_NAMES, TanhTransform


REF_SCALES = {
    "xvr_nd": 0.01,
    "coeff_drag_sway": 0.1,
    "cry_cross_flow": 0.5,
    "crn_cross_flow": 0.5,
    "coeff_drag_fore": 0.01,
    "coeff_drag_aft": 0.01,
    "tau_prop": 0.01,
    "coeff_cp_prop": 0.01,
    "ai_coeff_prop0": 0.001,
    "ai_coeff_prop1": 0.001,
    "ai_coeff_prop2": 0.001,
    "ai_coeff_prop3": 0.001,
    "ai_coeff_prop4": 0.001,
    "ai_coeff_prop5": 0.001,
    "ai_coeff_prop6": 0.001,
    "ai_coeff_prop7": 0.001,
    "bi_coeff_prop0": 0.001,
    "bi_coeff_prop1": 0.001,
    "bi_coeff_prop2": 0.001,
    "bi_coeff_prop3": 0.001,
    "bi_coeff_prop4": 0.001,
    "bi_coeff_prop5": 0.001,
    "bi_coeff_prop6": 0.001,
    "bi_coeff_prop7": 0.001,
    "ci_coeff_prop0": 0.01,
    "ci_coeff_prop1": 0.01,
    "ci_coeff_prop2": 0.01,
    "ci_coeff_prop3": 0.01,
    "ah_rudder": 0.1,
    "kx_rudder": 0.1,
    "kx_rudder_reverse": 0.1,
    "epsilon_rudder": 0.1,
    "cpr_rudder": 0.1,
    "gammaN_rudder": 0.1,
    "gammaP_rudder": 0.1,
}


def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def residual(theta_dict, batch, constants, vel_scale, pose_scale, lambda_pose):
    state_init, state_target, control, wind = batch
    pred = mmg.rollout(
        state_init,
        control,
        wind,
        theta_dict,
        constants,
        dt=dataset.DT,
        eps=mmg.SMOOTH_EPS_DEFAULT,
    )
    vel_idx = [1, 3, 5]
    vel = (pred[:, :, vel_idx] - state_target[:, :, vel_idx]) / vel_scale.view(1, 1, -1)
    pred_pose = torch.stack(
        [
            pred[:, :, 0] - state_init[:, None, 0],
            pred[:, :, 2] - state_init[:, None, 2],
            wrap_to_pi(pred[:, :, 4] - state_init[:, None, 4]),
        ],
        dim=-1,
    )
    target_pose = torch.stack(
        [
            state_target[:, :, 0] - state_init[:, None, 0],
            state_target[:, :, 2] - state_init[:, None, 2],
            wrap_to_pi(state_target[:, :, 4] - state_init[:, None, 4]),
        ],
        dim=-1,
    )
    pose = math.sqrt(lambda_pose) * (pred_pose - target_pose) / pose_scale.view(1, 1, -1)
    return torch.cat([vel.flatten(), pose.flatten()])


def alpha_scale(theta_star, transform):
    theta_efd = transform.theta_efd.cpu().numpy()
    return np.array(
        [
            max(abs(theta_star[i]), abs(theta_efd[i]), REF_SCALES.get(name, 0.01))
            for i, name in enumerate(PARAM_NAMES)
        ],
        dtype=np.float64,
    )


def jacobian_column(index, theta_star, scale, batch, constants, vel_scale, pose_scale, lambda_pose):
    theta_base = torch.tensor(theta_star, dtype=torch.float64)
    scale_tensor = torch.tensor(scale, dtype=torch.float64)

    def residual_from_alpha(alpha):
        theta = theta_base + scale_tensor * alpha
        theta_dict = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
        return residual(theta_dict, batch, constants, vel_scale, pose_scale, lambda_pose)

    alpha = torch.zeros(len(PARAM_NAMES), dtype=torch.float64)
    direction = torch.zeros_like(alpha)
    direction[index] = 1.0
    _, column = torch_func.jvp(residual_from_alpha, (alpha,), (direction,))
    return column.detach().cpu().numpy()


def trial_offsets(trajectories, segment_length=dataset.SEG_LEN):
    offsets = {}
    segment_index = 0
    for trial_index, trajectory in enumerate(trajectories):
        first = segment_index
        for start in range(0, len(trajectory["state"]) - segment_length, segment_length):
            if start + segment_length + 1 > len(trajectory["state"]):
                break
            segment_index += 1
        offsets[str(trial_index)] = [first, segment_index]
    return offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="results/full/diffmmg_seed0.json")
    parser.add_argument("--output-dir", default="results/rrqr")
    parser.add_argument("--data-root", default=dataset.DATA_ROOT)
    parser.add_argument("--lambda-pose", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    with open(args.anchor, "r", encoding="utf-8") as handle:
        anchor = json.load(handle)
    if anchor["param_names"] != PARAM_NAMES:
        raise ValueError("The anchor parameter order does not match PARAM_NAMES.")

    theta_star = np.asarray(anchor["theta_best"], dtype=np.float64)
    transform = TanhTransform()
    scale = alpha_scale(theta_star, transform)
    trajectories, _ = dataset.load_random_trajectories(args.data_root)
    if not trajectories:
        raise FileNotFoundError(f"No random-steering CSV files found under {args.data_root!r}.")
    segments = dataset.make_segments(trajectories)
    batch = dataset.segments_to_tensors(segments)
    vel_scale, pose_scale = dataset.compute_scales(batch[0], batch[1])
    constants = ShipConstants()

    row_count = residual(
        {name: torch.tensor(theta_star[i]) for i, name in enumerate(PARAM_NAMES)},
        batch,
        constants,
        vel_scale,
        pose_scale,
        args.lambda_pose,
    ).numel()
    jacobian = np.empty((row_count, len(PARAM_NAMES)), dtype=np.float64)
    started = time.perf_counter()
    for index, name in enumerate(PARAM_NAMES):
        jacobian[:, index] = jacobian_column(
            index,
            theta_star,
            scale,
            batch,
            constants,
            vel_scale,
            pose_scale,
            args.lambda_pose,
        )
        print(f"[{index + 1:02d}/{len(PARAM_NAMES)}] {name}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, "jacobian_alpha.npz"),
        J=jacobian,
        theta_star=theta_star,
        s=scale,
        param_names=np.asarray(PARAM_NAMES),
        lambda_pose=np.asarray(args.lambda_pose),
    )
    with open(os.path.join(args.output_dir, "jacobian_trial_offsets.json"), "w", encoding="utf-8") as handle:
        json.dump(trial_offsets(trajectories), handle, indent=2)
    elapsed = time.perf_counter() - started
    print(f"Saved {jacobian.shape} Jacobian to {args.output_dir} in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
