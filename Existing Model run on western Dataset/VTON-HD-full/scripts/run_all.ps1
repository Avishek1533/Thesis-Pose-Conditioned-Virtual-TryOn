Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ImageCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    return (Get-ChildItem -Path $Path -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png)$' } | Measure-Object).Count
}

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $repoRoot = Resolve-Path (Join-Path $scriptDir "..")
    Set-Location $repoRoot

    Write-Host "Ensuring scripts directory exists..."
    if (-not (Test-Path ".\scripts")) {
        New-Item -ItemType Directory ".\scripts" | Out-Null
    }

    Write-Host "Checking git status..."
    $gitStatus = & git "status" "--short"
    if ($LASTEXITCODE -ne 0) {
        throw "git status failed with exit code $LASTEXITCODE."
    }
    $restoreTargets = @()
    if ($gitStatus -match "test\.py") { $restoreTargets += "test.py" }
    if ($gitStatus -match "datasets\.py") { $restoreTargets += "datasets.py" }
    if ($restoreTargets.Count -gt 0) {
        $uniqueTargets = $restoreTargets | Select-Object -Unique
        Write-Host ("Restoring files: {0}" -f ($uniqueTargets -join ", "))
        & git "restore" "--source=HEAD" "--worktree" "--staged" @uniqueTargets
        if ($LASTEXITCODE -ne 0) {
            throw "git restore failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host "Bootstrapping environment..."
    & powershell "-NoProfile" "-ExecutionPolicy" "Bypass" "-File" (Join-Path $repoRoot "scripts\bootstrap_env.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Environment bootstrap failed with exit code $LASTEXITCODE."
    }

    Write-Host "Validating checkpoints..."
    $checkpoints = @("alias_final.pth", "gmm_final.pth", "seg_final.pth")
    foreach ($cp in $checkpoints) {
        $cpPath = Join-Path $repoRoot ("checkpoints\" + $cp)
        if (-not (Test-Path $cpPath)) {
            throw "Checkpoint missing: $cpPath"
        }
    }

    Write-Host "Validating dataset folders..."
    $requiredDirs = @(
        "train\image", "train\cloth", "train\cloth-mask",
        "test\image", "test\cloth", "test\cloth-mask"
    )
    foreach ($dir in $requiredDirs) {
        $fullPath = Join-Path $repoRoot ("datasets\" + $dir)
        if (-not (Test-Path $fullPath)) {
            throw "Missing dataset directory: $fullPath"
        }
    }

    Write-Host "Generating OpenPose-like data (resume)..."
    $genArgs = @("run", "-n", "vitonhd", "--no-capture-output", "python", ".\scripts\generate_openpose_like.py", "--split", "both", "--device", "auto", "--resume")
    & conda @genArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Pose generation failed with exit code $LASTEXITCODE."
    }

    $datasetsRoot = Join-Path $repoRoot "datasets"
    $testImageDir = Join-Path $datasetsRoot "test\image"
    $testPoseImgDir = Join-Path $datasetsRoot "test\openpose-img"
    $testPoseJsonDir = Join-Path $datasetsRoot "test\openpose-json"

    $testImageCount = Get-ImageCount $testImageDir
    $testPoseImgCount = (Get-ChildItem -Path $testPoseImgDir -Filter "*_rendered.png" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $testPoseJsonCount = (Get-ChildItem -Path $testPoseJsonDir -Filter "*_keypoints.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count

    if (($testPoseImgCount -lt $testImageCount) -or ($testPoseJsonCount -lt $testImageCount)) {
        Write-Host "Detected missing pose outputs; regenerating test split without resume..."
        $regenArgs = @("run", "-n", "vitonhd", "--no-capture-output", "python", ".\scripts\generate_openpose_like.py", "--split", "test", "--device", "auto")
        & conda @regenArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Pose regeneration failed with exit code $LASTEXITCODE."
        }
        $testPoseImgCount = (Get-ChildItem -Path $testPoseImgDir -Filter "*_rendered.png" -File -ErrorAction SilentlyContinue | Measure-Object).Count
        $testPoseJsonCount = (Get-ChildItem -Path $testPoseJsonDir -Filter "*_keypoints.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count
        if (($testPoseImgCount -lt $testImageCount) -or ($testPoseJsonCount -lt $testImageCount)) {
            throw "Pose generation did not produce enough files (images: $testImageCount, renders: $testPoseImgCount, json: $testPoseJsonCount)."
        }
    }

    Write-Host "Running inference..."
    $testArgs = @("run", "-n", "vitonhd", "--no-capture-output", "python", ".\test.py", "--name", "alias_final", "--workers", "0", "--load_height", "1024", "--load_width", "768")
    & conda @testArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Inference failed with exit code $LASTEXITCODE."
    }

    $resultsDir = Join-Path $repoRoot "results"
    $aliasDir = Join-Path $resultsDir "alias_final"
    if (-not (Test-Path $aliasDir)) {
        Write-Host "Inference completed, but results directory not found at $aliasDir"
    } else {
        $latest = Get-ChildItem -Path $aliasDir -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Host "Results directory: $aliasDir"
        if ($latest) {
            Write-Host ("Sample result: {0}" -f $latest.FullName)
        } else {
            Write-Host "No files found under results directory."
        }
    }

    Write-Host "Pipeline completed successfully."
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
