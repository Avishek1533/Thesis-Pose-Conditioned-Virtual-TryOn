import os, argparse, glob
from PIL import Image
import numpy as np
import pandas as pd
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_images(dir_path):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))
    return sorted(files)

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def ext_lower(path):
    return os.path.splitext(path)[1].lower()

def pick_best_path(paths, prefer_exts=(".jpg", ".png", ".jpeg", ".webp", ".bmp")):
    if len(paths) == 1:
        return paths[0]
    paths_by_ext = {ext_lower(p): p for p in paths}
    for e in prefer_exts:
        if e in paths_by_ext:
            return paths_by_ext[e]
    return paths[0]

def build_stem_map(files, prefer_exts=(".jpg", ".png", ".jpeg", ".webp", ".bmp")):
    bucket = {}
    for p in files:
        k = stem(p)
        bucket.setdefault(k, []).append(p)
    out = {}
    dups = 0
    for k, ps in bucket.items():
        if len(ps) > 1:
            dups += 1
        out[k] = pick_best_path(ps, prefer_exts=prefer_exts)
    return out, dups

def load_rgb(path):
    return Image.open(path).convert("RGB")

def to_np(img: Image.Image):
    return np.array(img).astype(np.float32) / 255.0  # [0,1]

def resize_to_min(a: Image.Image, b: Image.Image):
    if a.size == b.size:
        return a, b
    w = min(a.size[0], b.size[0])
    h = min(a.size[1], b.size[1])
    return a.resize((w, h), Image.BICUBIC), b.resize((w, h), Image.BICUBIC)

def compute_ssim(img_a: Image.Image, img_b: Image.Image):
    a = to_np(img_a)
    b = to_np(img_b)
    scores = []
    for c in range(3):
        scores.append(ssim(a[..., c], b[..., c], data_range=1.0))
    return float(np.mean(scores))

def pil_to_lpips_tensor(img: Image.Image, device):
    x = to_np(img)  # [0,1]
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    x = x * 2.0 - 1.0  # [-1,1]
    return x.to(device)

def compute_lpips(loss_fn, img_a: Image.Image, img_b: Image.Image, device):
    ta = pil_to_lpips_tensor(img_a, device)
    tb = pil_to_lpips_tensor(img_b, device)
    with torch.no_grad():
        d = loss_fn(ta, tb)
    return float(d.item())

def fake_to_real_key(fake_key: str, match_mode: str):
    """
    match_mode:
      - stem   : fake stem == real stem
      - person : fake = PERSON_CLOTH_00  -> real = PERSON_00
    """
    if match_mode == "stem":
        return fake_key
    if match_mode == "person":
        parts = fake_key.split("_")
        if len(parts) >= 2:
            person = parts[0]
            return f"{person}_00"
        return fake_key
    return fake_key

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--out_csv", default="metrics_per_image.csv")
    ap.add_argument("--resize_mode", choices=["min", "real"], default="min",
                    help="min: resize both to min size; real: resize fake to real size")
    ap.add_argument("--match_mode", choices=["stem", "person"], default="person",
                    help="stem: same filename; person: fake PERSON_CLOTH_00 -> real PERSON_00")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    real_files = list_images(args.real_dir)
    fake_files = list_images(args.fake_dir)

    # Choose preferred extension if duplicates exist
    real_map, real_dups = build_stem_map(real_files, prefer_exts=(".jpg", ".png", ".jpeg", ".webp", ".bmp"))
    fake_map, fake_dups = build_stem_map(fake_files, prefer_exts=(".jpg", ".png", ".jpeg", ".webp", ".bmp"))

    # Build pairs according to match_mode
    pairs = []
    missing_real = 0
    for fk, fp in fake_map.items():
        rk = fake_to_real_key(fk, args.match_mode)
        rp = real_map.get(rk, None)
        if rp is None:
            missing_real += 1
            continue
        pairs.append((fk, fp, rk, rp))

    if args.max_images and args.max_images > 0:
        pairs = pairs[:args.max_images]

    print(f"Real images total : {len(real_files)}  (duplicate stems: {real_dups})")
    print(f"Fake images total : {len(fake_files)}  (duplicate stems: {fake_dups})")
    print(f"Matched pairs     : {len(pairs)}  (match_mode={args.match_mode})")
    print(f"Missing REAL      : {missing_real}")

    if len(pairs) == 0:
        print("\n[!] No matched pairs found.")
        print("    Debug tip: print few names from each folder and verify naming pattern.")
        return

    device = torch.device(args.device)
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    rows = []
    for i, (fk, fp, rk, rp) in enumerate(pairs, 1):
        try:
            real_img = load_rgb(rp)
            fake_img = load_rgb(fp)

            if args.resize_mode == "min":
                real_img, fake_img = resize_to_min(real_img, fake_img)
            else:
                if fake_img.size != real_img.size:
                    fake_img = fake_img.resize(real_img.size, Image.BICUBIC)

            s = compute_ssim(real_img, fake_img)
            l = compute_lpips(loss_fn, real_img, fake_img, device)

            rows.append({
                "fake_key": fk,
                "real_key": rk,
                "fake_path": fp,
                "real_path": rp,
                "ssim": s,
                "lpips": l
            })

            if i % 100 == 0 or i == len(pairs):
                print(f"[{i}/{len(pairs)}] SSIM={s:.4f} LPIPS={l:.4f}")

        except Exception as e:
            rows.append({
                "fake_key": fk,
                "real_key": rk,
                "fake_path": fp,
                "real_path": rp,
                "ssim": np.nan,
                "lpips": np.nan,
                "error": str(e)
            })
            print(f"[WARN] Failed on {fk}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("\n---- SUMMARY ----")
    print(f"Saved CSV: {os.path.abspath(args.out_csv)}")
    print(f"SSIM mean : {df['ssim'].mean():.6f}")
    print(f"LPIPS mean: {df['lpips'].mean():.6f}")
    if "error" in df.columns:
        nerr = df["error"].notna().sum()
        if nerr:
            print(f"Errors    : {nerr} (see 'error' column)")

if __name__ == "__main__":
    main()
