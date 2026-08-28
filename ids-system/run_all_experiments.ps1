<#
.SYNOPSIS
    Runs SSSL-IDS Continual Learning Benchmark across 0%, 5%, and 10% Labeled Data.

.PARAMETER datasets
    List of datasets: "anoshift", "cicids2017", "kddcup99", "unsw" or "all" (default: "all")

.PARAMETER label_ratios
    List of label proportions: "0", "0.05", "0.10" (default: "0", "0.05", "0.10")

.PARAMETER quick
    Run in quick mode with small epoch count for testing (default: false)
#>

param (
    [string[]]$datasets = @("all"),
    [string[]]$label_ratios = @("0", "0.05", "0.10"),
    [switch]$quick
)

$ErrorActionPreference = "Stop"

# Ensure we are in the ids-system directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($null -ne $scriptDir -and $scriptDir -ne "") {
    Set-Location $scriptDir
}

# Check virtual environment
if ($null -eq $env:VIRTUAL_ENV) {
    if (Test-Path ".\.venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Cyan
        . .\.venv\Scripts\Activate.ps1
    }
}

$pyArgs = @("run_experiments.py", "--datasets") + $datasets + @("--label_ratios") + $label_ratios
if ($quick) {
    $pyArgs += "--quick"
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host " STARTING MULTI-DATASET FEW-LABEL SSSL-IDS BENCHMARK RUNNER" -ForegroundColor Green
Write-Host " Datasets:     $($datasets -join ' ')" -ForegroundColor Cyan
Write-Host " Label Ratios: $($label_ratios -join ' ')" -ForegroundColor Cyan
Write-Host " Quick Mode:   $quick" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green

python @pyArgs
