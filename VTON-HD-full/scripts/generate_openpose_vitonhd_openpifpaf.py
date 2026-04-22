#!/usr/bin/env python3
"""
Generate OpenPifPaf-style openpose outputs for VITON-HD test set.

Produces:
 - rendered pose PNG: <stem>_rendered.png  => datasets/test/openpose-img/
 - keypoints JSON:   <stem>_keypoints.json => datasets/test/openpose-json/

CLI:
 python generate_openpose_vitonhd_openpifpaf.py --src .\datasets\test\image --out_img_dir .\datasets\test\openpose-img --out_json_dir .\datasets\test\openpose-json --device cuda
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import sys

# COCO skeleton for drawing (pairs of keypoint indices)
COCO_SKELETON = [
    (0,1),(1,3),(0,2),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(11,12),(11,13),(13,15),(12,14),(14,16)
]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, image_name: str, keypoints):
    payload = {"image": image_name, "keypoints": keypoints}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def render_keypoints(image, keypoints):
    img = image.copy()
    h, w = img.shape[:2]
    # keypoints: list of [x,y,score] or empty
    for kp in keypoints:
        x, y, s = kp
        cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
    # draw skeleton using COCO indices if enough keypoints
    try:
        for a,b in COCO_SKELETON:
            if a < len(keypoints) and b < len(keypoints):
                xa, ya, sa = keypoints[a]
                xb, yb, sb = keypoints[b]
                if xa>0 and xb>0 and ya>0 and yb>0:
                    cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), (0,200,255), 2)
    except Exception:
        pass
    return img

def try_openpifpaf_predict(img_bgr, device):
    # Attempt to use OpenPifPaf Predictor API; be resilient to API variations.
    try:
        import openpifpaf
        Predictor = None
        try:
            from openpifpaf.predictor import Predictor as _P
            Predictor = _P
        except Exception:
            if hasattr(openpifpaf, "Predictor"):
                Predictor = openpifpaf.Predictor
        if Predictor is None:
            return None
        # instantiate predictor (best-effort)
        try:
            pred = Predictor(checkpoint=None, device=device)
        except TypeError:
            try:
                pred = Predictor(checkpoint=None, gpu=(device == "cuda"))
            except Exception:
                pred = Predictor()
        # Try multiple call styles
        try:
            annots, meta = pred.numpy_image(img_bgr)
        except Exception:
            try:
                annots = pred.annotations(img_bgr)
                meta = None
            except Exception:
                try:
                    annots, _, meta = pred.predict(img_bgr)
                except Exception:
                    return None
        # Convert first person annotation to keypoints list
        persons = []
        for a in annots:
            data = None
            if hasattr(a, "data"):
                data = np.array(a.data).reshape(-1,3)
            elif hasattr(a, "keypoints"):
                data = np.array(a.keypoints).reshape(-1,3)
            elif hasattr(a, "json_data"):
                data = np.array(a.json_data).reshape(-1,3)
            if data is None:
                continue
            persons.append(data.tolist())
        return persons
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=r".\datasets\test\image")
    p.add_argument("--out_img_dir", default=r".\datasets\test\openpose-img")
    p.add_argument("--out_json_dir", default=r".\datasets\test\openpose-json")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    args = p.parse_args()

    src = Path(args.src).resolve()
    out_img = Path(args.out_img_dir).resolve()
    out_json = Path(args.out_json_dir).resolve()

    if not src.exists() or not src.is_dir():
        print("Source folder not found:", src, file=sys.stderr)
        return 2

    exts = {".jpg",".jpeg",".png"}
    imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
    if not imgs:
        print("No images found in:", src, file=sys.stderr)
        return 3

    ensure_dir(out_img)
    ensure_dir(out_json)

    processed = 0
    for img_path in imgs:
        stem = img_path.stem
        out_img_path = out_img / f"{stem}_rendered.png"
        out_json_path = out_json / f"{stem}_keypoints.json"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # write empty files
            save_json(out_json_path, img_path.name, [])
            blank = 255 * np.ones((256,256,3), dtype=np.uint8)
            cv2.imwrite(str(out_img_path), blank)
            continue

        persons = try_openpifpaf_predict(img_bgr, args.device)
        if persons is None:
            # fallback: save empty
            keypoints = []
            rendered = img_bgr.copy()
        else:
            # take first person if exists
            if len(persons) >= 1:
                kp = persons[0]
                keypoints = [[float(x), float(y), float(s)] for (x,y,s) in kp]
                rendered = render_keypoints(img_bgr, keypoints)
            else:
                keypoints = []
                rendered = img_bgr.copy()

        save_json(out_json_path, img_path.name, keypoints)
        cv2.imwrite(str(out_img_path), rendered)
        processed += 1

    print(f"Processed {processed}/{len(imgs)} images.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
