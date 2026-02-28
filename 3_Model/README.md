# Model Definition and Evaluation

**[Notebook](model_definition_evaluation.ipynb)**

## Overview

This module contains the complete model training pipeline and evaluation analysis for the ROSE-CV ocean surface emulator. Four U-Net iterations are compared against a persistence baseline to forecast weekly Sea Surface Temperature (SST) over the Cape Verde region.

## Model Iterations

| # | Model | Key Change |
| :--- | :--- | :--- |
| 1 | **Base U-Net** | Standard encoder-decoder, direct state prediction |
| 2 | **Residual U-Net** | Predicts temperature delta instead of absolute state |
| 3 | **Boundary-Aware Residual** | Upweights land-mask boundary pixels in convolutions |
| 4 | **Deep Residual U-Net** | Additional encoder-decoder level for broader receptive field |


## Evaluation Utilities

[Eval.py](Eval.py) provides three helper functions used in the evaluation notebook:

- `plot_training_history(csv_path)` — step-wise and epoch-wise loss curves
- `display_final_metrics(csv_path)` — formatted test-set metrics table
- `plot_model_predictions(nc_path)` — side-by-side GT / prediction / error maps with cartopy

> **Note:** The U-Net backbone source code is not included due to licensing constraints. See [backbone_src/README.md](Pipeline/backbone_src/README.md) for details.
