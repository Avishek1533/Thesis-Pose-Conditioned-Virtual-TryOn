import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch

try:
    import lpips  # pylint: disable=import-outside-toplevel
except ImportError as exc:
    print("lpips package not found. Install with `pip install lpips` inside vitonhd environment.", file=sys.stderr)
    raise SystemExit(1) from exc


def gather_by_stem(path: Path, patterns: Tuple[str, ...]) -> Tuple[Dict[str, Path], List[str]]:
    mapping: Dict[str, Path] = {}
    warnings: List[str] = []
    for pattern in patterns:
        for file_path in sorted(path.glob(pattern)):
            stem = file_path.stem
            if stem in mapping:
                warnings.append(f"Duplicate detected for stem '{stem}': keeping '{mapping[stem]}' and ignoring '{file_path}'.")
                continue
            mapping[stem] = file_path
    return mapping, warnings


def resize_to_square(image: Image.Image, target: int) -> Image.Image:
    if image.size[0] == target and image.size[1] == target:
        return image
    return image.resize((target, target), resample=Image.BICUBIC)


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def compute_metrics(real_image: Image.Image, fake_image: Image.Image, lpips_model: lpips.LPIPS, device: torch.device) -> Tuple[float, float]:
    real_np = np.array(real_image).astype(np.float32) / 255.0
    fake_np = np.array(fake_image).astype(np.float32) / 255.0
    ssim_val = structural_similarity(real_np, fake_np, channel_axis=2, data_range=1.0)
    real_tensor = pil_to_tensor(real_image, device)
    fake_tensor = pil_to_tensor(fake_image, device)
    with torch.no_grad():
        lpips_val = float(lpips_model(real_tensor, fake_tensor).item())
    return ssim_val, lpips_val


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SSIM and LPIPS metrics by matching image stems.")
    parser.add_argument("--real_dir", default=str(Path("datasets") / "test" / "image"))
    parser.add_argument("--fake_dir", default=str(Path("results") / "alias_final" / "images"))
    parser.add_argument("--output_csv", default=str(Path("results") / "alias_final" / "evaluation" / "metrics" / "metrics_per_image.csv"))
    parser.add_argument("--summary_path", default=str(Path("results") / "alias_final" / "evaluation" / "metrics" / "summary.txt"))
    parser.add_argument("--patterns", nargs="+", default=["*.jpg", "*.jpeg", "*.png"])
    args = parser.parse_args()

    real_dir = Path(args.real_dir).resolve()
    fake_dir = Path(args.fake_dir).resolve()
    csv_path = Path(args.output_csv).resolve()
    summary_path = Path(args.summary_path).resolve()

    ensure_dir(csv_path.parent)

    if not real_dir.exists():
        print(f"Real directory missing: {real_dir}", file=sys.stderr)
        raise SystemExit(2)
    if not fake_dir.exists():
        print(f"Fake directory missing: {fake_dir}", file=sys.stderr)
        raise SystemExit(3)

    real_map, real_warnings = gather_by_stem(real_dir, tuple(args.patterns))
    fake_map, fake_warnings = gather_by_stem(fake_dir, tuple(args.patterns))
    for msg in real_warnings + fake_warnings:
        print(f"Warning: {msg}")

    stems = sorted(set(real_map.keys()) & set(fake_map.keys()))
    if not stems:
        print("No matching stems found between real and fake directories.", file=sys.stderr)
        raise SystemExit(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    records: List[Tuple[str, Path, Path, float, float]] = []
    missing_real = sorted(set(fake_map.keys()) - set(real_map.keys()))
    missing_fake = sorted(set(real_map.keys()) - set(fake_map.keys()))
    for stem in missing_real:
        print(f"Warning: Fake image stem '{stem}' has no matching real counterpart.")
    for stem in missing_fake:
        print(f"Warning: Real image stem '{stem}' has no matching fake counterpart.")

    for stem in stems:
        real_path = real_map[stem]
        fake_path = fake_map[stem]
        try:
            real_img = Image.open(real_path).convert("RGB")
            fake_img = Image.open(fake_path).convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Failed to open pair '{stem}' due to {exc}. Skipping.")
            continue

        min_dim = min(min(real_img.size), min(fake_img.size))
        if min_dim <= 0:
            print(f"Warning: Invalid dimensions for pair '{stem}'. Skipping.")
            continue

        real_resized = resize_to_square(real_img, min_dim)
        fake_resized = resize_to_square(fake_img, min_dim)

        try:
            ssim_val, lpips_val = compute_metrics(real_resized, fake_resized, lpips_model, device)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Metric computation failed for '{stem}' due to {exc}. Skipping.")
            continue

        records.append((stem, real_path, fake_path, ssim_val, lpips_val))

    if not records:
        print("No metrics computed; all pairs failed.", file=sys.stderr)
        raise SystemExit(5)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename_stem", "real_path", "fake_path", "ssim", "lpips"])
        for stem, real_path, fake_path, ssim_val, lpips_val in records:
            writer.writerow([stem, str(real_path), str(fake_path), f"{ssim_val:.6f}", f"{lpips_val:.6f}"])

    ssim_values = np.array([row[3] for row in records], dtype=np.float32)
    lpips_values = np.array([row[4] for row in records], dtype=np.float32)

    summary_lines = [
        f"Total pairs: {len(records)}",
        f"SSIM: mean={ssim_values.mean():.6f}, std={ssim_values.std():.6f}, min={ssim_values.min():.6f}, max={ssim_values.max():.6f}",
        f"LPIPS: mean={lpips_values.mean():.6f}, std={lpips_values.std():.6f}, min={lpips_values.min():.6f}, max={lpips_values.max():.6f}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Metrics written to {csv_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
