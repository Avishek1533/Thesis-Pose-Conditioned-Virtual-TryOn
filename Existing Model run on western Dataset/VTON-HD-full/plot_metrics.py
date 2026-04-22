import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def save_hist(series, title, out_path, bins=50):
    plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_box(df, cols, title, out_path):
    plt.figure()
    df[cols].plot(kind="box")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_cdf(series, title, out_path):
    x = series.dropna().sort_values().values
    if len(x) == 0:
        return
    y = (range(1, len(x)+1))
    y = [v/len(x) for v in y]
    plt.figure()
    plt.plot(x, y)
    plt.title(f"CDF: {title}")
    plt.xlabel(title)
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_scatter(x, y, title, out_path):
    plt.figure()
    plt.scatter(x, y, s=6)
    plt.title(title)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="metric_plots")
    ap.add_argument("--open_dir", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty. First generate metrics_per_image.csv with SSIM/LPIPS rows.")

    need = {"ssim","lpips"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"CSV must have columns {need}, but got: {list(df.columns)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Histograms
    save_hist(df["ssim"],  "SSIM",  os.path.join(args.out_dir, "hist_ssim.png"))
    save_hist(df["lpips"], "LPIPS", os.path.join(args.out_dir, "hist_lpips.png"))

    # Boxplot
    save_box(df, ["ssim","lpips"], "SSIM & LPIPS (Boxplot)", os.path.join(args.out_dir, "box_ssim_lpips.png"))

    # CDF
    save_cdf(df["ssim"],  "SSIM",  os.path.join(args.out_dir, "cdf_ssim.png"))
    save_cdf(df["lpips"], "LPIPS", os.path.join(args.out_dir, "cdf_lpips.png"))

    # Scatter (SSIM vs LPIPS)
    save_scatter(df["ssim"], df["lpips"], "SSIM vs LPIPS", os.path.join(args.out_dir, "scatter_ssim_vs_lpips.png"))

    # Quick summary txt
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"CSV: {os.path.abspath(args.csv)}\n")
        f.write(f"N: {len(df)}\n")
        f.write(f"SSIM mean:  {df.ssim.mean():.6f}\n")
        f.write(f"LPIPS mean: {df.lpips.mean():.6f}\n")

    print("Saved plots to:", os.path.abspath(args.out_dir))

    if args.open_dir:
        os.startfile(os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
