param(
  [string]$EnvName = "vitonhd",
  [string]$PreferredName = "alias_final"
)

$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[INFO] $m" }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Fail($m){ Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1 }

function Run-CondaPy([string]$args){
  & conda run -n $EnvName --no-capture-output python $args
  return $LASTEXITCODE
}

function Get-TorchInfo {
  $code = "import torch; print(torch.__version__); print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))"
  & conda run -n $EnvName --no-capture-output python -c $code
  return $LASTEXITCODE
}

function Patch-TestPy {
  Info "Patching test.py for CPU/GPU auto device (replace .cuda() with .to(device)) ..."
  if (-not (Test-Path ".\test.py")) { Fail "test.py not found in repo root." }

  $py = @"
import re, pathlib
p = pathlib.Path('test.py')
s = p.read_text(encoding='utf-8')

# inject device if missing
if 'device = torch.device(' not in s:
    s = s.replace(
        'def main():',
        'def main():\n    import torch\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    print(\"Using device:\", device)\n'
    )

# replace .cuda().eval() -> .to(device).eval()
s = re.sub(r'(\w+)\.cuda\(\)\.eval\(\)', r'\1.to(device).eval()', s)

# replace remaining .cuda() -> .to(device)
s = re.sub(r'(\w+)\.cuda\(\)', r'\1.to(device)', s)

p.write_text(s, encoding='utf-8')
print('PATCH_OK')
"@

  & conda run -n $EnvName --no-capture-output python -c $py
}

function Find-DatasetSrc {
  $candidates = @(
    "D:\Mohoshin thesis\VTON-HD\_downloads\vitonhd\extracted",
    "D:\Mohoshin thesis\_downloads\vitonhd\extracted",
    "D:\Mohoshin thesis\VTON-HD\_downloads\vitonhd\extracted\extracted",
    "D:\Mohoshin thesis\_downloads\vitonhd\extracted\extracted"
  )
  foreach($p in $candidates){
    if (Test-Path $p){ return $p }
  }
  return $null
}

function Ensure-Dataset {
  $src = Find-DatasetSrc
  if (-not $src) {
    Fail "Dataset extracted folder not found. Expected one of the common paths under D:\Mohoshin thesis\... (_downloads\vitonhd\extracted)."
  }

  Info "Dataset src found: $src"
  New-Item -ItemType Directory -Path ".\datasets" -Force | Out-Null

  if (-not (Test-Path ".\scripts\prepare_vitonhd_dataset.py")) {
    Fail "Missing scripts\prepare_vitonhd_dataset.py. Your scripts folder is incomplete."
  }

  Info "Preparing dataset into .\datasets ..."
  & conda run -n $EnvName --no-capture-output python ".\scripts\prepare_vitonhd_dataset.py" --src "$src" --dst ".\datasets"
}

function Ensure-Checkpoints {
  if (-not (Test-Path ".\checkpoints")) { Fail "Missing .\checkpoints folder." }

  if (Test-Path ".\checkpoints\checkpoints") {
    Info "Flattening nested checkpoints folder..."
    Move-Item ".\checkpoints\checkpoints\*" ".\checkpoints\" -Force
    Remove-Item ".\checkpoints\checkpoints" -Recurse -Force
  }

  $pth = Get-ChildItem ".\checkpoints" -File -Filter *.pth -ErrorAction SilentlyContinue
  if (-not $pth -or $pth.Count -lt 1) { Fail "No .pth files found in .\checkpoints" }

  Info "Checkpoints present:"
  $pth | ForEach-Object { Write-Host "  - $($_.Name)" }
}

function Pick-ExperimentName {
  if (Test-Path ".\checkpoints\$PreferredName.pth") { return $PreferredName }
  $first = (Get-ChildItem ".\checkpoints" -File -Filter *.pth | Select-Object -First 1).BaseName
  return $first
}

function Run-Inference([string]$name){
  Info "Running inference with --name $name"
  & conda run -n $EnvName --no-capture-output python ".\test.py" --name "$name"
  return $LASTEXITCODE
}

# ---------------- main ----------------
Info "Autopilot starting in: $(Get-Location)"
Ensure-Checkpoints
Ensure-Dataset

Info "Torch info:"
Get-TorchInfo | Out-Null

$name = Pick-ExperimentName
Info "Selected experiment name: $name"

# First try
$code = Run-Inference $name
if ($code -ne 0) {
  Warn "Inference failed (exit=$code). Trying auto-fix for CPU-only torch by patching test.py, then retry once..."
  Patch-TestPy
  $code2 = Run-Inference $name
  if ($code2 -ne 0) {
    Fail "Inference still failing after patch (exit=$code2). Check the error above."
  }
}

Info "Autopilot finished successfully."
