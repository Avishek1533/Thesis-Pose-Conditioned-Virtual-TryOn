import os
import glob
from PIL import Image
from tqdm import tqdm

import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips
import torchvision.transforms as T

# --------- paths (CHANGE IF NEEDED) ----------
ROOT = r"D:\Mohoshin thesis\VTON-HD-full"
GEN_DIR = os.path.join(ROOT, "results", "alias_final")     # generated .jpg
REAL_DIR = os.path.join(ROOT, "datasets", "test", "image") # ground-truth person images
PAIR_TXT = os.path.join(ROOT, "datasets", "test_pairs.txt")
# ---------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

to_tensor = T.Compose([
    T.ToTensor(),  # [0,1]
])

# SSIM (expects [0,1])
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# LPIPS (expects [-1,1])
lpips_model = lpips.LPIPS(net="alex").to(device)

def load_img(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return img

def img_to_t01(img):
    return to_tensor(img).unsqueeze(0)  # 1x3xHxW in [0,1]

def t01_to_t11(t01):
    return t01 * 2.0 - 1.0  # [-1,1]

# Build a mapping from generated filenames for fast lookup
gen_files = glob.glob(os.path.join(GEN_DIR, "*.jpg"))
gen_map = {os.path.basename(p): p for p in gen_files}

# Read pairs
pairs = []
with open(PAIR_TXT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        a, b = line.split()
        # a=person image name like 13878_00.jpg
        # b=cloth image name like 08143_00.jpg
        pairs.append((a, b))

ssim_vals = []
lpips_vals = []
missing = 0

for person_img, cloth_img in tqdm(pairs, desc="Evaluating"):
    person_id = os.path.splitext(person_img)[0]  # 13878_00
    cloth_id  = os.path.splitext(cloth_img)[0]   # 08143_00

    # Many repos save output like: 13878_08143_00.jpg
    # We'll try a couple of patterns:
    cand1 = f"{person_id.split('_')[0]}_{cloth_id}.jpg"       # 13878_08143_00.jpg
    cand2 = f"{person_id}_{cloth_id}.jpg"                    # 13878_00_08143_00.jpg (fallback)
    out_path = gen_map.get(cand1) or gen_map.get(cand2)

    if out_path is None:
        missing += 1
        continue

    gt_path = os.path.join(REAL_DIR, person_img)
    if not os.path.exists(gt_path):
        missing += 1
        continue

    # Load same size
    out_img = load_img(out_path)
    gt_img  = load_img(gt_path, size=out_img.size)

    out_t01 = img_to_t01(out_img).to(device)
    gt_t01  = img_to_t01(gt_img).to(device)

    # SSIM
    ssim = ssim_metric(out_t01, gt_t01).item()
    ssim_vals.append(ssim)

    # LPIPS
    out_t11 = t01_to_t11(out_t01)
    gt_t11  = t01_to_t11(gt_t01)
    d = lpips_model(out_t11, gt_t11).mean().item()
    lpips_vals.append(d)

# Report
def safe_mean(x):
    return sum(x) / len(x) if len(x) else float("nan")

print("---- RESULTS ----")
print("Pairs total:", len(pairs))
print("Used pairs :", len(ssim_vals))
print("Missing   :", missing)
print("SSIM mean :", safe_mean(ssim_vals))
print("LPIPS mean:", safe_mean(lpips_vals))
