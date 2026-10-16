from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.linalg import qr, svdvals


def sensitivities(jacobian):
    return np.linalg.norm(jacobian, axis=0) / np.sqrt(jacobian.shape[0])


def filter_columns(delta, threshold):
    maximum = float(delta.max())
    if maximum <= 0.0:
        raise ValueError("All Jacobian columns have zero sensitivity.")
    relative = delta / maximum
    kept = np.flatnonzero(relative >= threshold)
    if kept.size == 0:
        raise ValueError("No parameter passed the sensitivity threshold.")
    return kept, relative, maximum


def rrqr_order(jacobian):
    return qr(jacobian, mode="economic", pivoting=True)[2]


def gamma_curve(jacobian, pivots):
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("The filtered Jacobian contains a zero column.")
    ordered = (jacobian / norms[None, :])[:, pivots]
    triangular = qr(ordered, mode="economic", pivoting=False)[1]
    gamma = np.empty(len(pivots), dtype=np.float64)
    for size in range(1, len(pivots) + 1):
        minimum = svdvals(triangular[:size, :size])[-1]
        gamma[size - 1] = np.inf if minimum <= np.finfo(float).eps else 1.0 / minimum
    return gamma


def selected_size(gamma, threshold):
    accepted = np.flatnonzero(gamma < threshold)
    return int(accepted[-1] + 1) if accepted.size else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jacobian", default="results/rrqr/jacobian_alpha.npz")
    parser.add_argument("--output-dir", default="results/rrqr")
    parser.add_argument("--tau-delta", type=float, default=1e-4)
    parser.add_argument("--gamma-crit", type=float, default=20.0)
    parser.add_argument("--reference-gamma", type=float, nargs="*", default=[10.0, 15.0])
    args = parser.parse_args()

    archive = np.load(args.jacobian)
    jacobian = np.asarray(archive["J"], dtype=np.float64)
    names = [str(name) for name in archive["param_names"]]
    if jacobian.ndim != 2 or jacobian.shape[1] != len(names):
        raise ValueError("Jacobian dimensions and parameter names are inconsistent.")
    if not np.isfinite(jacobian).all():
        raise ValueError("Jacobian contains non-finite entries.")

    delta = sensitivities(jacobian)
    kept, relative, delta_max = filter_columns(delta, args.tau_delta)
    filtered = jacobian[:, kept]
    pivots = rrqr_order(filtered)
    gamma = gamma_curve(filtered, pivots)
    order_indices = kept[pivots]
    order_names = [names[index] for index in order_indices]
    primary_size = selected_size(gamma, args.gamma_crit)
    reference_sizes = {str(value): selected_size(gamma, value) for value in args.reference_gamma}

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(
        {
            "parameter": names,
            "delta": delta,
            "delta_relative": relative,
            "kept": np.isin(np.arange(len(names)), kept),
        }
    ).to_csv(os.path.join(args.output_dir, "sensitivity_table.csv"), index=False, float_format="%.12e")
    pd.DataFrame(
        {
            "K": np.arange(1, len(gamma) + 1),
            "gamma_K": gamma,
            "sigma_min": 1.0 / gamma,
        }
    ).to_csv(os.path.join(args.output_dir, "gamma_curve.csv"), index=False, float_format="%.12e")

    rrqr_result = {
        "filtered_indices": kept.tolist(),
        "ordered_indices": order_indices.tolist(),
        "ordered_parameters": order_names,
    }
    with open(os.path.join(args.output_dir, "rrqr_order.json"), "w", encoding="utf-8") as handle:
        json.dump(rrqr_result, handle, indent=2)

    excluded_by_sensitivity = [names[index] for index in range(len(names)) if index not in set(kept)]
    result = {
        "method": "RRQR-guided trajectory-sensitivity subset selection",
        "tau_delta": args.tau_delta,
        "gamma_crit": args.gamma_crit,
        "n_parameters": len(names),
        "n_after_sensitivity_filter": int(len(kept)),
        "K_star": primary_size,
        "gamma_at_K_star": float(gamma[primary_size - 1]) if primary_size else None,
        "selected_parameters": order_names[:primary_size],
        "excluded_parameters": excluded_by_sensitivity + order_names[primary_size:],
        "rrqr_order": order_names,
        "reference_subset_sizes": reference_sizes,
        "reference_subsets": {
            key: order_names[:size] for key, size in reference_sizes.items()
        },
    }
    with open(os.path.join(args.output_dir, "subset_primary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Sensitivity filter retained {len(kept)}/{len(names)} parameters.")
    print(f"RRQR selected K={primary_size} at gamma_crit={args.gamma_crit:g}.")
    for key, size in reference_sizes.items():
        print(f"Reference gamma={key}: K={size}.")
    print(f"Saved selection outputs to {args.output_dir}.")


if __name__ == "__main__":
    main()
