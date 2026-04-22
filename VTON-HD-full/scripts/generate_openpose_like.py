import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

try:
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNN_ResNet50_FPN_Weights
    _WEIGHTS = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
except Exception:  # pylint: disable=broad-except
    from torchvision.models.detection import keypointrcnn_resnet50_fpn  # type: ignore
    _WEIGHTS = None

OPENPOSE18_MAP = {
    0: 0,
    2: 6,
    3: 8,
    4: 10,
    5: 5,
    6: 7,
    7: 9,
    8: 12,
    9: 14,
    10: 16,
    11: 11,
    12: 13,
    13: 15,
    14: 2,
    15: 1,
    16: 4,
    17: 3,
}

BODY25_MAP = {
    0: 0,
    2: 6,
    3: 8,
    4: 10,
    5: 5,
    6: 7,
    7: 9,
    9: 12,
    10: 14,
    11: 16,
    12: 11,
    13: 13,
    14: 15,
    15: 2,
    16: 1,
    17: 4,
    18: 3,
}

OPENPOSE18_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (1, 8), (8, 11),
    (0, 14), (14, 16),
    (0, 15), (15, 17)
]

BODY25_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20), (20, 21),
    (11, 22), (22, 23), (23, 24)
]


def find_datasets_py(repo_root: Path) -> Path:
    primary = repo_root / "datasets.py"
    if primary.is_file():
        return primary
    for candidate in repo_root.rglob("datasets.py"):
        return candidate
    raise FileNotFoundError("Unable to locate datasets.py in repository.")


def infer_schema(datasets_py_path: Path) -> Tuple[str, int]:
    text = datasets_py_path.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"reshape\s*\(\s*25\s*,\s*3\s*\)", text) or "BODY_25" in text or re.search(r"25\s*\*\s*3", text):
        return "body25", 25
    if re.search(r"reshape\s*\(\s*18\s*,\s*3\s*\)", text):
        return "coco18", 18
    return "coco18", 18


def average_points(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    valid = [p for p in points if p[2] > 0]
    if not valid:
        return 0.0, 0.0, 0.0
    x = sum(p[0] for p in valid) / len(valid)
    y = sum(p[1] for p in valid) / len(valid)
    c = sum(p[2] for p in valid) / len(valid)
    return x, y, min(c, 1.0)


def clamp_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    result = points.copy()
    result[:, 0] = np.clip(result[:, 0], 0, max(width - 1, 0))
    result[:, 1] = np.clip(result[:, 1], 0, max(height - 1, 0))
    return result


def coco17_to_openpose18(keypoints: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    result = np.zeros((18, 3), dtype=np.float32)
    if keypoints is None:
        return result
    keypoints = clamp_points(keypoints, width, height)
    for dst_idx, src_idx in OPENPOSE18_MAP.items():
        result[dst_idx] = keypoints[src_idx]
    neck = average_points([tuple(keypoints[5]), tuple(keypoints[6])])
    result[1] = neck
    return result


def coco17_to_body25(keypoints: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    result = np.zeros((25, 3), dtype=np.float32)
    if keypoints is None:
        return result
    keypoints = clamp_points(keypoints, width, height)
    for dst_idx, src_idx in BODY25_MAP.items():
        result[dst_idx] = keypoints[src_idx]
    neck = average_points([tuple(keypoints[5]), tuple(keypoints[6])])
    result[1] = neck
    mid_hip = average_points([tuple(keypoints[11]), tuple(keypoints[12])])
    result[8] = mid_hip
    return result


def render_skeleton(points: np.ndarray, width: int, height: int, schema: str) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    threshold = 0.05
    pairs = OPENPOSE18_PAIRS if schema == "coco18" else BODY25_PAIRS
    for idx, (x, y, conf) in enumerate(points):
        if conf > threshold:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)
    for start, end in pairs:
        if start >= len(points) or end >= len(points):
            continue
        if points[start][2] > threshold and points[end][2] > threshold:
            pt1 = (int(round(points[start][0])), int(round(points[start][1])))
            pt2 = (int(round(points[end][0])), int(round(points[end][1])))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
    return canvas


def flatten_points(points: np.ndarray) -> List[float]:
    return [float(v) for v in points.reshape(-1)]


def load_model(device: torch.device) -> torch.nn.Module:
    if _WEIGHTS is not None:
        model = keypointrcnn_resnet50_fpn(weights=_WEIGHTS)
    else:
        model = keypointrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    model.eval()
    return model


def detect_keypoints(model: torch.nn.Module, device: torch.device, image_tensor: torch.Tensor) -> Tuple[Optional[np.ndarray], float]:
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])[0]
    if len(outputs["keypoints"]) == 0:
        return None, 0.0
    scores = outputs["scores"].detach().cpu().numpy()
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    keypoints = outputs["keypoints"][best_idx].detach().cpu().numpy()
    return keypoints, best_score


def collect_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        return []
    files = []
    for path in image_dir.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            files.append(path)
    files.sort()
    return files


def ensure_dirs(paths: List[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def process_split(split: str, datasets_root: Path, model: torch.nn.Module, device: torch.device, schema: str, max_images: Optional[int], resume: bool) -> Tuple[int, int, int]:
    image_dir = datasets_root / split / "image"
    pose_img_dir = datasets_root / split / "openpose-img"
    pose_json_dir = datasets_root / split / "openpose-json"
    ensure_dirs([pose_img_dir, pose_json_dir])

    images = collect_images(image_dir)
    if max_images is not None:
        images = images[:max_images]

    processed = 0
    skipped = 0
    generated = 0

    for idx, image_path in enumerate(images, start=1):
        stem = image_path.stem
        pose_png = pose_img_dir / f"{stem}_rendered.png"
        pose_json = pose_json_dir / f"{stem}_keypoints.json"

        if resume and pose_png.is_file() and pose_json.is_file():
            skipped += 1
            continue

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        width, height = pil_image.size
        tensor = to_tensor(pil_image)
        keypoints, det_score = detect_keypoints(model, device, tensor)
        if det_score < 0.15:
            keypoints = None

        if schema == "body25":
            points = coco17_to_body25(keypoints, width, height)
        else:
            points = coco17_to_openpose18(keypoints, width, height)
        render = render_skeleton(points, width, height, schema)

        payload = {
            "version": 1.3,
            "people": [
                {
                    "pose_keypoints_2d": flatten_points(points),
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": []
                }
            ]
        }

        pose_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        cv2.imwrite(str(pose_png), render)

        processed += 1
        generated += 1

        if idx % 100 == 0:
            print(f"[{split}] processed {idx}/{len(images)}")

    return len(images), processed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenPose-like keypoints using Keypoint R-CNN.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--datasets_root", default=str(Path(".") / "datasets"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    datasets_py_path = find_datasets_py(repo_root)
    schema, count = infer_schema(datasets_py_path)
    print(f"Detected schema {schema} with {count} keypoints from {datasets_py_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    splits = []
    if args.split in {"train", "both"}:
        splits.append("train")
    if args.split in {"test", "both"}:
        splits.append("test")

    total_images = 0
    total_processed = 0
    total_skipped = 0

    for split in splits:
        images, processed, skipped = process_split(
            split=split,
            datasets_root=Path(args.datasets_root),
            model=model,
            device=device,
            schema=schema,
            max_images=args.max_images,
            resume=args.resume,
        )
        print(f"[{split}] total images {images}, generated {processed}, skipped {skipped}")
        total_images += images
        total_processed += processed
        total_skipped += skipped

    print(f"Generation finished. Images: {total_images}, generated: {total_processed}, skipped: {total_skipped}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
