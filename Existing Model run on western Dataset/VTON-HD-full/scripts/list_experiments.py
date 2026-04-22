#!/usr/bin/env python3
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
ckpt_dir = repo_root / "checkpoints"

def detect_experiments():
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return []
    entries = sorted(ckpt_dir.iterdir())
    candidates = []
    for e in entries:
        if e.name == ".gitignore":
            continue
        if e.is_dir():
            candidates.append(e.name)
        elif e.is_file() and re.search(r'\.(pth|pt|pkl|tar|ckpt)$', e.name, re.I):
            name = re.sub(r'\.(pth|pt|pkl|tar|ckpt)$', '', e.name, flags=re.I)
            candidates.append(name)
    seen = set(); out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c); out.append(c)
    return out

def main():
    exps = detect_experiments()
    if not exps:
        print("No experiments found in ./checkpoints/")
        return 1
    print("Detected experiment candidates:")
    for e in exps:
        print("  -", e)
    print()
    print('Suggested name to use:', exps[0])
    print('Example PowerShell:')
    print(r'  .\scripts\run_test.ps1 -Name "' + exps[0] + '" -GPUId 0')
    return 0

if __name__ == "__main__":
    sys.exit(main())
