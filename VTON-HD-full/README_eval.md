# Evaluation Pipeline

## Directory Layout
- `datasets/train` and `datasets/test`: input datasets.
- `results/alias_final/images`: generated outputs paired with real images.
- `results/alias_final/logs`: log files from inference or evaluation.
- `results/alias_final/evaluation/metrics`: CSV and summary statistics.
- `results/alias_final/evaluation/plots`: metric visualizations.
- `results/alias_final/evaluation/fid`: FID text reports.
- `scripts`: evaluation helper scripts.

## Running Evaluation
Execute:
```
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_all_eval.ps1"
```
The script enforces the folder layout, relocates stray images/logs, runs metric computation, generates plots, and records FID with `pytorch_fid`.

## Metrics
- **SSIM** (higher is better): structural similarity between real and generated images.
- **LPIPS** (lower is better): perceptual distance using deep features.
- **FID** (lower is better): distribution distance between real and generated sets.

## Outputs
- Per-image metrics: `results/alias_final/evaluation/metrics/metrics_per_image.csv`
- Summary statistics: `results/alias_final/evaluation/metrics/summary.txt`
- Plots: `results/alias_final/evaluation/plots`
- FID results: `results/alias_final/evaluation/fid/fid.txt`
- Logs: `results/alias_final/logs`
