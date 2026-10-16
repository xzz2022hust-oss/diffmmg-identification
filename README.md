# DiffMMG: Gradient-Based Calibration and RRQR-Guided Parameter Subset Selection for MMG Maneuvering Models

Code for the manuscript:

> Z. Xu, Y. Ma, L. Jian, and X. Wang. *DiffMMG: Gradient-Based Calibration and RRQR-Guided Parameter Subset Selection for MMG Maneuvering Models*. Journal of Marine Engineering & Technology (under review).

DiffMMG implements the full MMG maneuvering model, including the four-quadrant propeller model, astern conditions, cross-flow integration, wind loads, and RK4 time integration, as an almost-everywhere differentiable PyTorch graph. Reverse-mode automatic differentiation calibrates the model parameters. Forward-mode Jacobian-vector products construct the parameter-to-trajectory Jacobian used for sensitivity screening and rank-revealing QR (RRQR) subset selection.

## Repository structure

| Path | Purpose |
|---|---|
| `mmg_model.py` | Differentiable hull, propeller, rudder, wind, and equations-of-motion model |
| `ship_constants.py` | Tier-1 constants loaded from the Esso Osaka input files |
| `transform.py` | Bounded tanh mapping between latent and physical parameters |
| `data.py` | FRT-DS-ESSO loading and non-overlapping shooting segments |
| `loss.py` | Normalized velocity and pose multiple-shooting loss |
| `diffmmg.py` | Full 45-parameter Adam calibration |
| `build_jacobian.py` | Forward-mode JVP construction of the weighted trajectory Jacobian |
| `subset_selection.py` | Sensitivity screening, RRQR ordering, and collinearity-index selection |
| `train_selected_subset.py` | Calibration of the selected parameters with all other parameters fixed at EFD values |
| `evaluate_berthing.py` | Held-out berthing evaluation at fixed horizons |
| `results/paper/` | Published RRQR order, sensitivity table, and collinearity curve |
| `results/revision/` | JSON results added during peer-review revision: correlations, trial-level cross-validation, baselines, and KCS-start robustness |

## Installation

The paper experiments used Python 3.9.18 and the pinned packages in `requirements.txt`. A GPU is not required.

```bash
git clone https://github.com/xzz2022hust-oss/diffmmg-identification.git
cd diffmmg-identification
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data and fixed MMG inputs

The free-running data are the Osaka University [FRT-DS-ESSO dataset](https://github.com/NAOE-5thLab/FRT-DS-ESSO). The fixed Esso Osaka model inputs are from [mmg-1p1r-simulator](https://github.com/NAOE-5thLab/mmg-1p1r-simulator). Clone both repositories beside this repository or under an `external` directory:

```bash
git clone https://github.com/NAOE-5thLab/FRT-DS-ESSO.git external/FRT-DS-ESSO
git clone https://github.com/NAOE-5thLab/mmg-1p1r-simulator.git external/mmg-1p1r-simulator
```

Set the two input locations before running the scripts. In PowerShell:

```powershell
$env:DIFFMMG_DATA_ROOT = (Resolve-Path "external/FRT-DS-ESSO/data")
$env:DIFFMMG_INPUT_CSV_DIR = (Resolve-Path "external/mmg-1p1r-simulator/mmg_module/input_csv")
```

In Bash:

```bash
export DIFFMMG_DATA_ROOT="$PWD/external/FRT-DS-ESSO/data"
export DIFFMMG_INPUT_CSV_DIR="$PWD/external/mmg-1p1r-simulator/mmg_module/input_csv"
```

The identification set is formed from the nine random-steering trials as 1,311 non-overlapping 50-step segments at `dt = 0.1 s`. The eight berthing trials are used only for held-out evaluation. The external repositories and their data remain subject to their own licenses.

## Reproduce the paper workflow

Run all commands from the repository root.

### 1. Full 45-parameter calibration

```bash
python diffmmg.py --n_steps 1000 --lr 1e-3 --seed 0 --threads 4 --save_dir results/full
```

This creates `results/full/diffmmg_seed0.json`.

### 2. Trajectory Jacobian and RRQR subset selection

```bash
python build_jacobian.py --anchor results/full/diffmmg_seed0.json --output-dir results/rrqr --threads 4
python subset_selection.py --jacobian results/rrqr/jacobian_alpha.npz --output-dir results/rrqr --tau-delta 1e-4 --gamma-crit 20 --reference-gamma 10 15
```

The first command creates the physical-coordinate Jacobian with forward-mode JVPs. The second command applies the relative sensitivity threshold, performs column-pivoted QR on the retained raw columns, and evaluates the Brun collinearity index on column-normalized nested subsets. With the paper settings, `gamma_crit = 20` gives `K = 27`; the stricter reference threshold `gamma_crit = 15` gives `K = 17`.

### 3. Calibrate the K = 27 and K = 17 subsets

```bash
python train_selected_subset.py --selection results/rrqr/subset_primary.json --k 27 --n-steps 1000 --lr 1e-3 --seed 0 --threads 4 --output-dir results/subset
python train_selected_subset.py --selection results/rrqr/subset_primary.json --k 17 --n-steps 1000 --lr 1e-3 --seed 0 --threads 4 --output-dir results/subset
```

Only the selected parameters are optimized. Every excluded parameter is reset to its EFD value after each optimizer update.

### 4. Evaluate the held-out berthing trials

```bash
python evaluate_berthing.py --model "DiffMMG-45=results/full/diffmmg_seed0.json" --model "DiffMMG-K27=results/subset/diffmmg_K27_seed0.json" --model "DiffMMG-K17=results/subset/diffmmg_K17_seed0.json" --threads 4 --output-dir results/berthing
```

This writes per-window results to `results/berthing/berthing_windows.csv` and horizon summaries to `results/berthing/berthing_summary.csv`.

## Reproducibility record

| Item | Paper setting |
|---|---|
| Repository version | `v1.0.3` |
| Python | 3.9.18 |
| PyTorch | 2.3.1, CPU, `float64` |
| NumPy / pandas / SciPy | 1.26.4 / 2.3.3 / 1.13.1 |
| Hardware | AMD Ryzen 9 7945HX, 4 CPU threads |
| Optimizer | Adam on latent coordinates |
| Calibration settings | learning rate `1e-3`, 1,000 updates, gradient-norm clipping at 10, pose weight `lambda = 0.3` |
| Primary random seed | 0 for NumPy and PyTorch |
| RRQR randomness | None; JVP construction, sensitivity screening, and pivoted QR are deterministic for a fixed anchor and environment |
| Segment construction | Deterministic, non-overlapping 50-step segments |

Package versions can be checked with:

```bash
python -c "import numpy, pandas, scipy, torch; print(torch.__version__, numpy.__version__, pandas.__version__, scipy.__version__)"
```

## License

The source code in this repository is released under the [MIT License](LICENSE). The FRT-DS-ESSO data and the upstream MMG input files are not included and remain governed by their respective upstream licenses.
