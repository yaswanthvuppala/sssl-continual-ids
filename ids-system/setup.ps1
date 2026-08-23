<#
.SYNOPSIS
    First-time setup: creates venv and installs dependencies.
    Run this ONCE before training.
#>
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Setting up SSSL-Based IDS..." -ForegroundColor Cyan

# Create virtual environment
if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate
. .\.venv\Scripts\Activate.ps1
Write-Host "Virtual environment activated." -ForegroundColor Green

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

Write-Host "`nSetup complete! Now run one of:" -ForegroundColor Green
Write-Host "  .\run_kddcup99.ps1    (for KDD Cup 99)" -ForegroundColor Cyan
Write-Host "  .\run_cicids2017.ps1  (for CICIDS 2017)" -ForegroundColor Cyan
