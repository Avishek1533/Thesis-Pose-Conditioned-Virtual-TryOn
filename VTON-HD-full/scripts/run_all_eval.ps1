Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $repoRoot = Resolve-Path (Join-Path $scriptDir "..")
    Set-Location $repoRoot

    $dirs = @(
        "datasets\train",
        "datasets\test",
        "checkpoints\alias_final",
        "results\alias_final",
        "results\alias_final\images",
        "results\alias_final\logs",
        "results\alias_final\evaluation",
        "results\alias_final\evaluation\fid",
        "results\alias_final\evaluation\metrics",
        "results\alias_final\evaluation\plots",
        "scripts"
    )
    foreach ($dir in $dirs) {
        Ensure-Directory (Join-Path $repoRoot $dir)
    }

    $imageSrc = Join-Path $repoRoot "results\alias_final"
    $imageDst = Join-Path $repoRoot "results\alias_final\images"
    if (Test-Path $imageSrc) {
        Get-ChildItem -Path $imageSrc -Filter "*.jpg" -File -ErrorAction SilentlyContinue | ForEach-Object {
            Move-Item -Path $_.FullName -Destination $imageDst -Force
        }
        Get-ChildItem -Path $imageSrc -Filter "*.jpeg" -File -ErrorAction SilentlyContinue | ForEach-Object {
            Move-Item -Path $_.FullName -Destination $imageDst -Force
        }
        Get-ChildItem -Path $imageSrc -Filter "*.png" -File -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_.DirectoryName -ne (Resolve-Path $imageDst).Path) {
                Move-Item -Path $_.FullName -Destination $imageDst -Force
            }
        }
    }

    $logDst = Join-Path $repoRoot "results\alias_final\logs"
    if (Test-Path $imageSrc) {
        Get-ChildItem -Path $imageSrc -File -ErrorAction SilentlyContinue | Where-Object {
            $_.Extension -in @(".log", ".txt") -or $_.Name -match "log"
        } | ForEach-Object {
            Move-Item -Path $_.FullName -Destination $logDst -Force
        }
    }

    $condaBase = @("run", "-n", "vitonhd", "--no-capture-output", "python")

    Write-Host "Running compute_metrics.py..."
    & conda @condaBase ".\scripts\compute_metrics.py"
    if ($LASTEXITCODE -ne 0) {
        throw "compute_metrics.py failed with exit code $LASTEXITCODE"
    }

    Write-Host "Running plot_metrics.py..."
    & conda @condaBase ".\scripts\plot_metrics.py"
    if ($LASTEXITCODE -ne 0) {
        throw "plot_metrics.py failed with exit code $LASTEXITCODE"
    }

    $realDir = Join-Path $repoRoot "datasets\test\image"
    $fakeDir = Join-Path $repoRoot "results\alias_final\images"
    $fidArgs = @("run", "-n", "vitonhd", "python", "-m", "pytorch_fid", $realDir, $fakeDir)

    Write-Host "Running FID..."
    $fidOutput = & conda @fidArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "pytorch_fid failed with exit code $LASTEXITCODE`n$fidOutput"
    }

    $fidPath = Join-Path $repoRoot "results\alias_final\evaluation\fid\fid.txt"
    $fidPath | Split-Path | Ensure-Directory
    Set-Content -Path $fidPath -Value ($fidOutput -join [Environment]::NewLine) -Encoding UTF8
    Write-Host "FID results saved to $fidPath"

    Write-Host "Evaluation pipeline completed."
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
