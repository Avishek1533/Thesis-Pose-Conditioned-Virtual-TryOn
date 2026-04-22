param(
  [string]$EnvName="vitonhd",
  [string]$Name="alias_final"
)
$ErrorActionPreference="Stop"
function Info($m){Write-Host "[INFO] $m"}
function Warn($m){Write-Host "[WARN] $m" -ForegroundColor Yellow}
function Fail($m){Write-Host "[FAIL] $m" -ForegroundColor Red; exit 1}

# sanity
if(-not (Test-Path ".\datasets\test")){ Fail "datasets\test not found. Run dataset prep first." }
if(-not (Test-Path ".\checkpoints\$Name.pth")){ Fail "Checkpoint not found: .\checkpoints\$Name.pth" }

$missing = "datasets\test\openpose-img\01780_00_rendered.png"
if (Test-Path $missing) {
  Info "openpose-img looks OK (sample exists): $missing"
} else {
  Warn "Missing sample openpose file: $missing"
  Info "Searching for the file name anywhere under .\datasets ..."
  $hit = Get-ChildItem -Path ".\datasets" -Recurse -Filter "01780_00_rendered.png" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($hit) {
    Info "Found it at: $($hit.FullName)"
    New-Item -ItemType Directory -Path ".\datasets\test\openpose-img" -Force | Out-Null
    Copy-Item -Path $hit.FullName -Destination ".\datasets\test\openpose-img\01780_00_rendered.png" -Force
    Info "Copied into expected location."
  } else {
    Warn "Exact file not found. Trying to locate any folder containing *_rendered.png ..."
    $cand = Get-ChildItem -Path ".\datasets\test" -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName.ToLower().Contains("openpose") -and ($_.FullName.ToLower().Contains("img") -or $_.FullName.ToLower().Contains("render")) } |
            Select-Object -First 1
    if (-not $cand) {
      Fail "No openpose rendered folder found under datasets\test. Your dataset is missing openpose-img. You need to generate openpose outputs."
    }
    Info "Candidate folder: $($cand.FullName)"
    $rendered = Get-ChildItem -Path $cand.FullName -Filter "*_rendered.png" -File -ErrorAction SilentlyContinue
    if (-not $rendered -or $rendered.Count -lt 10) {
      Fail "Found a candidate folder but it has too few *_rendered.png files. Dataset openpose outputs missing/incomplete."
    }
    New-Item -ItemType Directory -Path ".\datasets\test\openpose-img" -Force | Out-Null
    $copied = 0
    foreach($f in $rendered){
      $dst = Join-Path ".\datasets\test\openpose-img" $f.Name
      if(-not (Test-Path $dst)){
        Copy-Item $f.FullName $dst -Force
        $copied++
      }
    }
    Info "Copied $copied rendered openpose images into datasets\test\openpose-img"
  }
}

# quick counts
$cnt = (Get-ChildItem ".\datasets\test\openpose-img" -Filter "*.png" -File -ErrorAction SilentlyContinue | Measure-Object).Count
Info "openpose-img png count (test): $cnt"
if ($cnt -lt 1000) { Warn "Count seems low; inference may still fail on another missing file." }

# run inference
Info "Running inference (workers=0 to avoid Windows dataloader issues)..."
& conda run -n $EnvName --no-capture-output python ".\test.py" --name $Name --load_height 1024 --load_width 768 --workers 0
