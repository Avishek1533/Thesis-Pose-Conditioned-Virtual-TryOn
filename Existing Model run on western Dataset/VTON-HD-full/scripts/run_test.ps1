param(
    [Parameter(Mandatory=$true)]
    [string] $Name,
    [int] $GPUId = 0,
    [string] $CondaEnv = "vitonhd"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

# Ensure checkpoints nesting fixed
Write-Host "Ensuring checkpoints are flattened..."
& powershell -ExecutionPolicy Bypass -File .\scripts\fix_checkpoints_nesting.ps1

# Verify datasets
if (-not (Test-Path ".\datasets\train") -or -not (Test-Path ".\datasets\test")) {
    Write-Host "Datasets not found or incomplete under .\datasets. Run prepare_vitonhd_dataset.py first." -ForegroundColor Red
    exit 2
}

# Inspect test.py help inside conda env to detect flags
Write-Host "Inspecting test.py --help..."
$helpText = & conda run -n $CondaEnv --no-capture-output python .\test.py --help 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: running help failed with conda run; trying local python --help..."
    $helpText = & python .\test.py --help 2>&1
}

$useDataroot = $false
$useGpuIds = $false
if ($helpText -match "--dataroot") { $useDataroot = $true }
if ($helpText -match "--gpu_ids") { $useGpuIds = $true }
if ($helpText -match "--gpu") { $useGpuIds = $true }

# Build args
$argsList = @(".\test.py", "--name", $Name)
if ($useDataroot) { $argsList += @("--dataroot", ".\datasets") } elseif ($helpText -match "--dataset_root") { $argsList += @("--dataset_root", ".\datasets") }
if ($useGpuIds) { $argsList += @("--gpu_ids", $GPUId.ToString()) } elseif ($helpText -match "--gpu") { $argsList += @("--gpu", $GPUId.ToString()) }

$cmd = "conda run -n $CondaEnv --no-capture-output python " + ($argsList -join " ")
Write-Host "COMMAND:" $cmd

# Execute
cmd.exe /c $cmd
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Host "test.py exited with code $rc" -ForegroundColor Red
    exit $rc
}

Write-Host "Inference finished. Check repo for results (common folders: results/, checkpoints/, output/)."
exit 0
