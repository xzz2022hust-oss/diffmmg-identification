# Revision result data

This directory contains the machine-readable numerical results added during peer-review revision. All result tables and matrices in this directory use JSON; the raw third-party trajectory data are not redistributed.

| File | Contents | Manuscript location |
|---|---|---|
| `correlation_44.json` | Complete 44 × 44 parameter-correlation matrix after removal of the zero-sensitivity direction | Appendix C |
| `correlation_K27.json` | Complete 27 × 27 correlation matrix for the RRQR K=27 subset | Appendix C |
| `correlation_K17.json` | Complete 17 × 17 correlation matrix for the RRQR K=17 subset | Appendix C |
| `fold_table.json` | Nine leave-one-trial-out folds, fold-specific subsets, losses, and 60 s MSE values | Appendix D |
| `kcs_params.json` | EFD and KCS starting coefficients, calibrated values, constraints, and bounds | Appendix B |

## Conventions

- Correlation matrices use the exact order in `parameter_order`; `matrix[i][j]` corresponds to parameters `i` and `j`. They are local linearized correlations obtained from a cluster-robust sandwich covariance grouped by shooting segment.
- Losses are dimensionless. Endpoint-position mean squared errors are in m².
- In `fold_table.json`, `Sub15-fold` and `Sub20-fold` denote the subsets selected using `gamma_crit = 15` and `gamma_crit = 20`; their sizes can vary across folds.
- In `kcs_params.json`, the 23 coefficients are nondimensional MMG coefficients. `A_lock` starts from KCS values and `B_lock` starts from EFD values; both use the same four physics-based negative-sign constraints.

The external FRT-DS-ESSO trajectories and upstream MMG input files remain governed by their source repositories and licenses.
