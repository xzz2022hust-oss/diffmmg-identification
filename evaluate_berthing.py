from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from mmg_model import rollout
from ship_constants import ShipConstants
from transform import PARAM_NAMES, TanhTransform


DT = 0.1


def load_theta(path):
    with open(path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    return dict(zip(result["param_names"], result["theta_best"]))


def efd_theta():
    transform = TanhTransform()
    return {name: float(transform.theta_efd[index]) for index, name in enumerate(PARAM_NAMES)}


def tensor_parameters(values):
    return {name: torch.tensor(value, dtype=torch.float64) for name, value in values.items()}


def windows(frame, horizons):
    times = frame.index.to_numpy(dtype=float)
    result = []
    for horizon in horizons:
        start = times[0]
        while start + horizon <= times[-1] + 1e-6:
            indices = np.flatnonzero((times >= start) & (times <= start + horizon + 1e-6))
            if len(indices) >= 10:
                result.append((indices[0], indices[-1] + 1, horizon))
            start += horizon
    return result


def predict(frame, parameters, constants):
    if len(frame) < 2:
        raise ValueError("A berthing window must contain at least two samples.")
    state = torch.tensor(
        [[
            frame.iloc[0]["x_position_mid [m]"],
            frame.iloc[0]["u_velo [m/s]"],
            frame.iloc[0]["y_position_mid [m]"],
            frame.iloc[0]["vm_velo [m/s]"],
            frame.iloc[0]["psi_hat [rad]"],
            frame.iloc[0]["r_angvelo [rad/s]"],
        ]],
        dtype=torch.float64,
    )
    control = torch.tensor(
        frame.iloc[:-1][["delta_rudder [rad]", "n_prop [rps]"]].to_numpy(),
        dtype=torch.float64,
    ).unsqueeze(0)
    wind = torch.tensor(
        frame.iloc[:-1][["wind_velo_relative_mid [m/s]", "wind_dir_relative_mid [rad]"]].to_numpy(),
        dtype=torch.float64,
    ).unsqueeze(0)
    with torch.no_grad():
        return rollout(state, control, wind, parameters, constants, dt=DT)[0].cpu().numpy()


def metrics(frame, prediction):
    target = frame.iloc[1:]
    if len(prediction) != len(target):
        raise ValueError("Prediction and target lengths are inconsistent.")
    return {
        "u_mse": float(np.mean((prediction[:, 1] - target["u_velo [m/s]"].to_numpy()) ** 2)),
        "vm_mse": float(np.mean((prediction[:, 3] - target["vm_velo [m/s]"].to_numpy()) ** 2)),
        "r_mse": float(np.mean((prediction[:, 5] - target["r_angvelo [rad/s]"].to_numpy()) ** 2)),
        "trajectory_rmse": float(
            np.sqrt(
                np.mean(
                    (prediction[:, 0] - target["x_position_mid [m]"].to_numpy()) ** 2
                    + (prediction[:, 2] - target["y_position_mid [m]"].to_numpy()) ** 2
                )
            )
        ),
    }


def parse_models(specifications):
    models = {"EFD": efd_theta()}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("Each --model value must use LABEL=PATH.")
        label, path = specification.split("=", 1)
        models[label] = load_theta(path)
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DIFFMMG_DATA_ROOT", "data"))
    parser.add_argument("--horizons", type=float, nargs="+", default=[10.0, 30.0, 60.0])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", default="results/berthing")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    models = {label: tensor_parameters(values) for label, values in parse_models(args.model).items()}
    constants = ShipConstants()
    files = sorted(glob.glob(os.path.join(args.data_root, "berthing", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No berthing CSV files found under {args.data_root!r}.")

    rows = []
    for path in files:
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.Index(pd.to_numeric(frame.index), name=frame.index.name)
        frame.columns = frame.columns.str.strip()
        for window_index, (first, last, horizon) in enumerate(windows(frame, args.horizons), start=1):
            section = frame.iloc[first:last]
            row = {"file": os.path.basename(path), "window": window_index, "horizon_s": horizon}
            for label, parameters in models.items():
                for metric, value in metrics(section, predict(section, parameters, constants)).items():
                    row[f"{label}_{metric}"] = value
            rows.append(row)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for horizon in args.horizons:
        subset = detail[detail["horizon_s"] == horizon]
        row = {"horizon_s": horizon, "n_windows": len(subset)}
        for label in models:
            for metric in ["trajectory_rmse", "u_mse", "vm_mse", "r_mse"]:
                values = subset[f"{label}_{metric}"]
                row[f"{label}_{metric}_mean"] = float(values.mean())
                row[f"{label}_{metric}_std"] = float(values.std())
        summary_rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    detail.to_csv(os.path.join(args.output_dir, "berthing_windows.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, "berthing_summary.csv"), index=False)
    print(f"Evaluated {len(files)} trials and {len(detail)} windows.")
    print(f"Saved evaluation outputs to {args.output_dir}.")


if __name__ == "__main__":
    main()
