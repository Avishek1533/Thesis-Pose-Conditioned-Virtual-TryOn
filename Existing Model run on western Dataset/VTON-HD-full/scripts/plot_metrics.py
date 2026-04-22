import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metrics(csv_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    stems: List[str] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                stems.append(row["filename_stem"])
                ssim_vals.append(float(row["ssim"]))
                lpips_vals.append(float(row["lpips"]))
            except KeyError:
                raise ValueError("CSV missing required columns: filename_stem, ssim, lpips")
            except ValueError:
                print(f"Warning: Skipping malformed row for stem '{row.get('filename_stem', '')}'.")
    if not stems:
        raise ValueError("No valid metric rows found.")
    return stems, np.array(ssim_vals, dtype=np.float32), np.array(lpips_vals, dtype=np.float32)


def save_histogram(values: np.ndarray, title: str, xlabel: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color="#1f77b4", alpha=0.8, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_boxplot(values: np.ndarray, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(4, 6))
    plt.boxplot(values, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#ff7f0e", color="#ff7f0e"),
                medianprops=dict(color="black"))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_cdf(values: np.ndarray, title: str, xlabel: str, path: Path) -> None:
    sorted_vals = np.sort(values)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_vals, cumulative, color="#2ca02c")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative Probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_scatter(x_vals: np.ndarray, y_vals: np.ndarray, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(x_vals, y_vals, s=15, alpha=0.7, color="#d62728")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SSIM and LPIPS metrics.")
    parser.add_argument("--metrics_csv", default=str(Path("results") / "alias_final" / "evaluation" / "metrics" / "metrics_per_image.csv"))
    parser.add_argument("--output_dir", default=str(Path("results") / "alias_final" / "evaluation" / "plots"))
    args = parser.parse_args()

    csv_path = Path(args.metrics_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    _, ssim_vals, lpips_vals = load_metrics(csv_path)

    save_histogram(ssim_vals, "SSIM Distribution", "SSIM", output_dir / "ssim_hist.png")
    save_histogram(lpips_vals, "LPIPS Distribution", "LPIPS", output_dir / "lpips_hist.png")
    save_boxplot(ssim_vals, "SSIM Boxplot", "SSIM", output_dir / "ssim_box.png")
    save_boxplot(lpips_vals, "LPIPS Boxplot", "LPIPS", output_dir / "lpips_box.png")
    save_cdf(ssim_vals, "SSIM CDF", "SSIM", output_dir / "cdf_ssim.png")
    save_cdf(lpips_vals, "LPIPS CDF", "LPIPS", output_dir / "cdf_lpips.png")
    save_scatter(ssim_vals, lpips_vals, "SSIM vs LPIPS", "SSIM (higher better)", "LPIPS (lower better)", output_dir / "scatter_ssim_vs_lpips.png")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
