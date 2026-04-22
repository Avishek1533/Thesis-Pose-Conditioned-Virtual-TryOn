#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
import sys

IMAGE_VARIANTS = {"image","images","img","imgs","image_highres"}
CLOTH_VARIANTS = {"cloth","cloths","clothes","garment","garments"}
MASK_VARIANTS = {"cloth-mask","cloth_mask","mask","masks","clothmask","mask_cloth"}

def find_pairs(root):
    train_p = None
    test_p = None
    for p in root.rglob("*.txt"):
        name = p.name.lower()
        if "train" in name and "pair" in name:
            train_p = p
        if "test" in name and "pair" in name:
            test_p = p
    return train_p, test_p

def find_split_root_by_subdirs(root):
    candidates = []
    for d in root.iterdir():
        if not d.is_dir(): continue
        subdirs = {x.name.lower() for x in d.iterdir() if x.is_dir()}
        if subdirs & IMAGE_VARIANTS and subdirs & CLOTH_VARIANTS and subdirs & MASK_VARIANTS:
            candidates.append(d)
    return candidates

def safe_copy_dir(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        s = item
        d = dst / item.name
        if s.is_dir():
            try:
                shutil.copytree(s, d)
            except FileExistsError:
                # merge
                for sub in s.iterdir():
                    if (d / sub.name).exists():
                        continue
                    try:
                        if sub.is_dir():
                            shutil.copytree(sub, d / sub.name)
                        else:
                            shutil.copy2(sub, d / sub.name)
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            try:
                shutil.copy2(s, d)
            except Exception:
                pass

def count_files(p):
    if not p.exists(): return 0
    return sum(1 for _ in p.iterdir() if _.is_file())

def read_first_lines(path, n=3):
    if not path or not path.exists(): return []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for _ in range(n):
                l = f.readline()
                if not l: break
                lines.append(l.rstrip('\n'))
            return lines
    except Exception:
        return []

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()

    src = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src.exists():
        print("ERROR: src not found:", src); sys.exit(2)

    # try to find train_pairs/test_pairs first
    train_p, test_p = find_pairs(src)
    # If found, prefer their parent as dataset root (if both in same parent)
    dataset_root = None
    if train_p and test_p and train_p.parent == test_p.parent:
        dataset_root = train_p.parent
    elif train_p and not test_p:
        dataset_root = train_p.parent
    elif test_p and not train_p:
        dataset_root = test_p.parent

    # fallback: look for top-level dirs containing image/cloth/mask
    if not dataset_root:
        candidates = find_split_root_by_subdirs(src)
        if candidates:
            # if two candidates found, try to pick those named train/test
            byname = {c.name.lower(): c for c in candidates}
            if "train" in byname and "test" in byname:
                dataset_root = src
            else:
                dataset_root = src

    if not dataset_root:
        # fallback to src itself
        dataset_root = src

    # Now identify train and test directories under dataset_root
    train_dir = None
    test_dir = None
    for d in dataset_root.iterdir():
        if not d.is_dir(): continue
        ln = d.name.lower()
        if "train" in ln and not train_dir: train_dir = d
        if "test" in ln and not test_dir: test_dir = d
    # if not found, try children that have image/cloth/mask
    if not train_dir or not test_dir:
        for d in dataset_root.iterdir():
            if not d.is_dir(): continue
            subdirs = {x.name.lower() for x in d.iterdir() if x.is_dir()}
            if subdirs & IMAGE_VARIANTS and subdirs & CLOTH_VARIANTS and subdirs & MASK_VARIANTS:
                if not train_dir:
                    train_dir = d
                elif not test_dir:
                    test_dir = d

    if not train_dir or not test_dir:
        print("ERROR: Could not find both train and test directories under", dataset_root)
        sys.exit(3)

    # prepare dst structure
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for split, src_split in (("train", train_dir), ("test", test_dir)):
        for key, variants in (("image", IMAGE_VARIANTS), ("cloth", CLOTH_VARIANTS), ("cloth-mask", MASK_VARIANTS)):
            dst_sub = dst / split / key
            # find matching subdir in src_split
            found = None
            for s in src_split.iterdir():
                if s.is_dir() and s.name.lower() in variants:
                    found = s
                    break
            if not found:
                # try any dir that startswith variant
                for s in src_split.iterdir():
                    if s.is_dir():
                        lname = s.name.lower()
                        for v in variants:
                            if lname.startswith(v) or v.startswith(lname):
                                found = s
                                break
                        if found: break
            if not found:
                print(f"WARNING: no source dir for {split}/{key} under {src_split}")
                continue
            safe_copy_dir(found, dst_sub)

    # copy pair files if found earlier
    if train_p:
        shutil.copy2(str(train_p), str(dst / "train_pairs.txt"))
    else:
        # try to find any train pair txt
        for p in src.rglob("*.txt"):
            if "train" in p.name.lower() and "pair" in p.name.lower():
                shutil.copy2(str(p), str(dst / "train_pairs.txt"))
                train_p = p
                break
    if test_p:
        shutil.copy2(str(test_p), str(dst / "test_pairs.txt"))
    else:
        for p in src.rglob("*.txt"):
            if "test" in p.name.lower() and "pair" in p.name.lower():
                shutil.copy2(str(p), str(dst / "test_pairs.txt"))
                test_p = p
                break

    # print summary
    print("Prepared dataset at:", dst)
    print("train - images:", count_files(dst / "train" / "image"), "cloth:", count_files(dst / "train" / "cloth"), "masks:", count_files(dst / "train" / "cloth-mask"))
    print("test  - images:", count_files(dst / "test" / "image"), "cloth:", count_files(dst / "test" / "cloth"), "masks:", count_files(dst / "test" / "cloth-mask"))
    print()
    print("train_pairs.txt (first 3 lines):")
    for l in read_first_lines(dst / "train_pairs.txt", 3):
        print(" ", l)
    print()
    print("test_pairs.txt (first 3 lines):")
    for l in read_first_lines(dst / "test_pairs.txt", 3):
        print(" ", l)

if __name__ == "__main__":
    main()
