$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) "..")
Set-Location $repoRoot
$ck = Join-Path $repoRoot "checkpoints"
$inner = Join-Path $ck "checkpoints"

if (Test-Path $inner) {
    Write-Host "Found nested folder:" $inner
    Get-ChildItem -Path $inner -Force | ForEach-Object {
        $dest = Join-Path $ck $_.Name
        if (Test-Path $dest) { Write-Host "Skipping existing item:" $_.Name; return }
        try {
            Move-Item -Path $_.FullName -Destination $dest -Force
            Write-Host "Moved:" $_.Name
        } catch {
            Write-Host "Move failed for:" $_.Name -ForegroundColor Yellow
            try { Copy-Item -Path $_.FullName -Destination $dest -Recurse -Force } catch { Write-Host "Copy also failed for:" $_.Name -ForegroundColor Red }
        }
    }
    try {
        Remove-Item -Path $inner -Force -Recurse
        Write-Host "Removed inner folder:" $inner
    } catch {
        Write-Host "Could not remove inner folder; check permissions." -ForegroundColor Yellow
    }
} else {
    Write-Host "No nested checkpoints folder found."
}

Write-Host ""
Write-Host "Final items in .\checkpoints (excluding .gitignore):"
Get-ChildItem -Path $ck -Force | Where-Object { $_.Name -ne ".gitignore" } | ForEach-Object { Write-Host "  - " $_.Name }
