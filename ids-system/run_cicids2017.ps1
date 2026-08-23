<#
.SYNOPSIS
    Run CICIDS 2017 training pipeline (SSL + 3 continual tasks + eval + transfer)
.NOTES
    Place your CICIDS 2017 dataset in: ..\CICIDS2017\
    Expected: 8 CSV files (Monday-WorkingHours.pcap_ISCX.csv, etc.)
#>
param (
    [string]$DataPath = "..\CICIDS2017",
    [int]$SSLEpochs = 20,
    [int]$IntrusionEpochs = 20,
    [int]$DoSEpochs = 15,
    [int]$PortScanEpochs = 15,
    [float]$EncoderLR = 0.003,
    [switch]$SkipSSL = $false
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate venv if present
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "[WARN] No .venv found. Using system python. Run 'python -m venv .venv' and 'pip install -r requirements.txt' first." -ForegroundColor Yellow
}

# Verify dataset exists
if (-not (Test-Path $DataPath)) {
    Write-Host "ERROR: CICIDS 2017 dataset not found at: $DataPath" -ForegroundColor Red
    Write-Host "Download from: https://www.unb.ca/cic/datasets/ids-2017.html" -ForegroundColor Yellow
    Write-Host "Place CSV files in: $DataPath" -ForegroundColor Yellow
    exit 1
}

$ds = "cicids2017"
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  CICIDS 2017 Training Pipeline" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Clean old outputs
foreach ($dir in @("checkpoints/$ds", "logs/$ds")) {
    if (Test-Path $dir) {
        Remove-Item -Path $dir -Recurse -Force
        Write-Host "Cleaned $dir" -ForegroundColor Yellow
    }
}

# Step 1: SSL
if (-not $SkipSSL) {
    Write-Host "`n[1/6] SSL Pretraining ($SSLEpochs epochs)..." -ForegroundColor Cyan
    python training/train_ssl.py --dataset cicids2017 --data_path $DataPath --label_col Label --epochs $SSLEpochs --dataset_name $ds
    if ($LASTEXITCODE -ne 0) { Write-Host "SSL FAILED" -ForegroundColor Red; exit 1 }
}

# Step 2: Intrusion
Write-Host "`n[2/6] Intrusion ($IntrusionEpochs epochs)..." -ForegroundColor Cyan
python training/train_task.py --task intrusion --dataset cicids2017 --data_path $DataPath --label_col Label --epochs $IntrusionEpochs --dataset_name $ds --unfreeze_encoder --encoder_lr $EncoderLR
if ($LASTEXITCODE -ne 0) { Write-Host "INTRUSION FAILED" -ForegroundColor Red; exit 1 }

# Step 3: DoS
Write-Host "`n[3/6] DoS ($DoSEpochs epochs)..." -ForegroundColor Cyan
python training/train_task.py --task dos --dataset cicids2017 --data_path $DataPath --label_col AttackCategory --epochs $DoSEpochs --dataset_name $ds --unfreeze_encoder --encoder_lr $EncoderLR
if ($LASTEXITCODE -ne 0) { Write-Host "DOS FAILED" -ForegroundColor Red; exit 1 }

# Step 4: Port Scan
Write-Host "`n[4/6] Port Scan ($PortScanEpochs epochs)..." -ForegroundColor Cyan
python training/train_task.py --task port_scan --dataset cicids2017 --data_path $DataPath --label_col AttackCategory --epochs $PortScanEpochs --dataset_name $ds --unfreeze_encoder --encoder_lr $EncoderLR
if ($LASTEXITCODE -ne 0) { Write-Host "PORT_SCAN FAILED" -ForegroundColor Red; exit 1 }

# Step 5: Evaluate
Write-Host "`n[5/6] Evaluation..." -ForegroundColor Cyan
python training/evaluate.py --task all --dataset cicids2017 --data_path $DataPath --label_col Label --dataset_name $ds
python training/visualize_metrics.py --task intrusion --dataset_name $ds
python training/visualize_metrics.py --task dos --dataset_name $ds
python training/visualize_metrics.py --task port_scan --dataset_name $ds

# Step 6: Transfer
Write-Host "`n[6/6] Transfer Matrix..." -ForegroundColor Cyan
python training/compute_transfer.py --dataset_name $ds --unfrozen

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  CICIDS 2017 COMPLETE!" -ForegroundColor Green
Write-Host "  Results in: logs/$ds/" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
