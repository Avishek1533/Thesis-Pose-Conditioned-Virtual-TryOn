#!/usr/bin/env python3
"""
Generate lightweight OpenPose-style outputs for VITON-HD test set using MediaPipe Pose.
Creates rendered skeleton PNGs and minimal keypoints JSON files.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def process_image(img_path: Path, out_img_path: Path, out_json_path: Path, pose):
    img = cv2.imread(str(img_path))
    if img is None:
        print("Could not read image:", img_path)
        return False
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x = float(lm.x) * w
            y = float(lm.y) * h
            # MediaPipe may provide visibility/score; fallback to z if available
            score = float(getattr(lm, "visibility", 1.0))
            keypoints.append([x, y, score])
    else:
        keypoints = []

    # Save JSON
    payload = {"image": img_path.name, "keypoints": keypoints}
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Render simple skeleton
    rendered = img.copy()
    if results.pose_landmarks:
        # draw landmarks
        for kp in keypoints:
            cx, cy = int(kp[0]), int(kp[1])
            cv2.circle(rendered, (cx, cy), 3, (0, 255, 0), -1)
        # draw connections using MediaPipe POSE_CONNECTIONS if available
        try:
            import mediapipe as mp
            for (a, b) in mp.solutions.pose.POSE_CONNECTIONS:
                a_lm = results.pose_landmarks.landmark[a]
                b_lm = results.pose_landmarks.landmark[b]
                ax, ay = int(a_lm.x * w), int(a_lm.y * h)
                bx, by = int(b_lm.x * w), int(b_lm.y * h)
                cv2.line(rendered, (ax, ay), (bx, by), (0, 200, 255), 2)
        except Exception:
            pass
    # ensure output dir
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), rendered)
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=r".\datasets\test\image", help="Source test images directory")
    p.add_argument("--out_img_dir", default=r".\datasets\test\openpose-img", help="Rendered pose images output dir")
    p.add_argument("--out_json_dir", default=r".\datasets\test\openpose-json", help="Keypoints JSON output dir")
    args = p.parse_args()

    src = Path(args.src).resolve()
    out_img_dir = Path(args.out_img_dir).resolve()
    out_json_dir = Path(args.out_json_dir).resolve()

    if not src.exists() or not src.is_dir():
        print("Source images folder not found:", src)
        return 2

    # gather image files
    exts = [".jpg", ".jpeg", ".png"]
    imgs = [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts]
    if not imgs:
        print("No test images found in:", src)
        return 3

    # initialize MediaPipe Pose
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.3)
    except Exception as e:
        print("Failed to import mediapipe:", e)
        print("Install with: pip install mediapipe")
        return 4

    success = 0
    for img_path in imgs:
        base = img_path.stem
        rendered_name = f"{base}_rendered.png"
        json_name = f"{base}_keypoints.json"
        out_img_path = out_img_dir / rendered_name
        out_json_path = out_json_dir / json_name
        ok = process_image(img_path, out_img_path, out_json_path, pose)
        if ok:
            success += 1

    pose.close()
    print(f"Processed {success}/{len(imgs)} images.")
    print("Rendered images dir:", out_img_dir)
    print("Keypoints json dir:", out_json_dir)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
