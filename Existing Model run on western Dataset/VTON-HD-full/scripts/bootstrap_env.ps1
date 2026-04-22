param(
    [string]$EnvName = "vitonhd"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $repoRoot = Resolve-Path (Join-Path $scriptDir "..")
    Set-Location $repoRoot

    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "conda executable not found in PATH."
    }

    $envJson = & conda "env" "list" "--json"
    if ($LASTEXITCODE -ne 0) {
        throw "conda env list failed with exit code $LASTEXITCODE."
    }

    $envData = $envJson | ConvertFrom-Json
    $envExists = $false
    foreach ($envPath in $envData.envs) {
        if ((Split-Path $envPath -Leaf) -ieq $EnvName) {
            $envExists = $true
            break
        }
    }

    if (-not $envExists) {
        Write-Host "Creating conda environment $EnvName..."
        & conda "create" "-y" "-n" $EnvName "python=3.8"
        if ($LASTEXITCODE -ne 0) {
            throw "conda create failed with exit code $LASTEXITCODE."
        }
    } else {
        Write-Host "Conda environment $EnvName already present."
    }

    $condaRun = @("run", "-n", $EnvName, "--no-capture-output")

    Write-Host "Upgrading pip..."
    & conda @condaRun "python" "-m" "pip" "install" "--upgrade" "pip"
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed with exit code $LASTEXITCODE."
    }

    Write-Host "Removing existing torch packages if present..."
    & conda @condaRun "python" "-m" "pip" "uninstall" "-y" "torch" "torchvision"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip uninstall returned $LASTEXITCODE, continuing."
        $global:LASTEXITCODE = 0
    }

    Write-Host "Installing torch and torchvision CUDA 11.8 wheels..."
    & conda @condaRun "python" "-m" "pip" "install" "torch" "torchvision" "--index-url" "https://download.pytorch.org/whl/cu118"
    if ($LASTEXITCODE -ne 0) {
        throw "torch installation failed with exit code $LASTEXITCODE."
    }

    Write-Host "Installing project dependencies..."
    $deps = @("numpy==1.24.4", "pillow", "opencv-python", "tqdm", "pyyaml", "scipy", "scikit-image")
    & conda @condaRun "python" "-m" "pip" "install" @deps
    if ($LASTEXITCODE -ne 0) {
        throw "dependency installation failed with exit code $LASTEXITCODE."
    }

    Write-Host "Torch diagnostics:"
    & conda @condaRun "python" "-c" "import torch; print('torch ' + torch.__version__); print('torch.cuda.is_available() ' + str(torch.cuda.is_available()))"
    if ($LASTEXITCODE -ne 0) {
        throw "torch diagnostics failed with exit code $LASTEXITCODE."
    }
    Write-Host "Environment bootstrap complete."
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
