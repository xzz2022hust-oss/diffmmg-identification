from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import data as dataset
import loss
from ship_constants import ShipConstants
from transform import PARAM_NAMES, TanhTransform


def set_threads(count):
    torch.set_num_threads(int(count))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.environ["OMP_NUM_THREADS"] = str(count)
    os.environ["MKL_NUM_THREADS"] = str(count)


def run(selected, n_steps, learning_rate, seed, threads, grad_clip, lambda_pose, save_path, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_threads(threads)

    constants = ShipConstants()
    transform = TanhTransform()
    trajectories, _ = dataset.load_random_trajectories()
    if not trajectories:
        raise FileNotFoundError(f"No random-steering CSV files found under {dataset.DATA_ROOT!r}.")
    segments = dataset.make_segments(trajectories)
    batch = dataset.segments_to_tensors(segments)
    vel_scale, pose_scale = dataset.compute_scales(batch[0], batch[1])

    unknown = sorted(set(selected) - set(PARAM_NAMES))
    if unknown:
        raise ValueError(f"Unknown parameter names: {unknown}")
    free_mask = torch.tensor([name in set(selected) for name in PARAM_NAMES], dtype=torch.bool)
    z_efd = transform.z_efd().detach()
    z = z_efd.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    history = []
    best_loss = float("inf")
    best_z = z.detach().clone()
    started = time.perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad()
        objective = loss.trajectory_loss(
            z,
            transform,
            batch,
            constants,
            vel_scale,
            pose_scale,
            lambda_pose=lambda_pose,
        )
        value = float(objective.detach())
        if value < best_loss:
            best_loss = value
            best_z = z.detach().clone()
        objective.backward()
        with torch.no_grad():
            z.grad[~free_mask] = 0.0
        torch.nn.utils.clip_grad_norm_([z], grad_clip)
        optimizer.step()
        with torch.no_grad():
            z[~free_mask] = z_efd[~free_mask]

        elapsed = time.perf_counter() - started
        history.append({"n_grad": step + 1, "loss": value, "wall_s": elapsed})
        if step % 50 == 0 or step + 1 == n_steps:
            print(f"[{tag} {step + 1:04d}/{n_steps}] loss={value:.6f} best={best_loss:.6f}", flush=True)

    with torch.no_grad():
        final_loss = float(
            loss.trajectory_loss(
                z,
                transform,
                batch,
                constants,
                vel_scale,
                pose_scale,
                lambda_pose=lambda_pose,
            )
        )
    if final_loss < best_loss:
        best_loss = final_loss
        best_z = z.detach().clone()

    theta_best = transform.to_theta(best_z).detach()
    fixed_deviation = (
        float((theta_best[~free_mask] - transform.theta_efd[~free_mask]).abs().max())
        if (~free_mask).any()
        else 0.0
    )
    result = {
        "optimizer": "adam_on_z_masked",
        "tag": tag,
        "seed": seed,
        "n_grad": n_steps,
        "lr": learning_rate,
        "grad_clip": grad_clip,
        "lambda_pose": lambda_pose,
        "threads": threads,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "wall_s": time.perf_counter() - started,
        "n_free_params": int(free_mask.sum()),
        "selected_parameters": selected,
        "excluded_fixed_at_efd": [name for name in PARAM_NAMES if name not in set(selected)],
        "excluded_max_deviation_from_efd": fixed_deviation,
        "param_names": PARAM_NAMES,
        "theta_best": theta_best.cpu().numpy().tolist(),
        "history": history,
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved {save_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="results/rrqr/subset_primary.json")
    parser.add_argument("--k", type=int)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--lambda-pose", type=float, default=0.3)
    parser.add_argument("--output-dir", default="results/subset")
    parser.add_argument("--tag")
    args = parser.parse_args()

    with open(args.selection, "r", encoding="utf-8") as handle:
        selection = json.load(handle)
    order = selection.get("rrqr_order", selection["selected_parameters"])
    size = args.k if args.k is not None else int(selection["K_star"])
    if size < 1 or size > len(order):
        raise ValueError(f"K must be between 1 and {len(order)}.")
    selected = order[:size]
    tag = args.tag or f"K{size}"
    save_path = os.path.join(args.output_dir, f"diffmmg_{tag}_seed{args.seed}.json")
    run(
        selected,
        args.n_steps,
        args.lr,
        args.seed,
        args.threads,
        args.grad_clip,
        args.lambda_pose,
        save_path,
        tag,
    )


if __name__ == "__main__":
    main()
