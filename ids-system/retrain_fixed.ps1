<#
.SYNOPSIS
    Retrain KDD Cup 99 and CICIDS 2017 with all 3 bug fixes applied.
    Run from: c:\Users\vuppa\Desktop\SSSL_Based_IDS\ids-system
.DESCRIPTION
    Fixes applied:
    1. Gradual lambda_u ramp (fixmatch_trainer.py)
    2. Temperature bound 50.0 (calibration.py)
    3. Stratified KDD test split (dataset_loader.py)
#>
param (
    [string[]]$datasets = @("kddcup99", "cicids2017"),
    [switch]$SkipSSL = $false
)

$ErrorActionPreference = "Stop"
Set-Location "c:\Users\vuppa\Desktop\SSSL_Based_IDS\ids-system"

# Activate venv
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated." -ForegroundColor Green
}

$encoder_lr = 0.003

foreach ($dataset in $datasets) {
    Write-Host "`n============================================" -ForegroundColor Green
    Write-Host "  RETRAINING: $dataset (with fixes)" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green

    # Clean old checkpoints and logs
    foreach ($dir in @("checkpoints/$dataset", "logs/$dataset")) {
        if (Test-Path $dir) {
            Remove-Item -Path $dir -Recurse -Force
            Write-Host "Cleaned $dir" -ForegroundColor Yellow
        }
    }

    if ($dataset -eq "kddcup99") {
        $DATA_PATH = "..\KDDCUP99"

        if (-not $SkipSSL) {
            Write-Host "`n[1/6] SSL Pretraining (20 epochs)..." -ForegroundColor Cyan
            python training/train_ssl.py --dataset kddcup99 --data_path $DATA_PATH --label_col Label --epochs 20 --dataset_name kddcup99
        }

        Write-Host "`n[2/6] Intrusion Task (30 epochs)..." -ForegroundColor Cyan
        python training/train_task.py --task intrusion --dataset kddcup99 --data_path $DATA_PATH --label_col Label --epochs 30 --dataset_name kddcup99 --unfreeze_encoder --encoder_lr $encoder_lr

        Write-Host "`n[3/6] DoS Task (25 epochs)..." -ForegroundColor Cyan
        python training/train_task.py --task dos --dataset kddcup99 --data_path $DATA_PATH --label_col AttackCategory --epochs 25 --dataset_name kddcup99 --unfreeze_encoder --encoder_lr $encoder_lr

        Write-Host "`n[4/6] Port Scan Task (25 epochs)..." -ForegroundColor Cyan
        python training/train_task.py --task port_scan --dataset kddcup99 --data_path $DATA_PATH --label_col AttackCategory --epochs 25 --dataset_name kddcup99 --unfreeze_encoder --encoder_lr $encoder_lr

        Write-Host "`n[5/6] Evaluation..." -ForegroundColor Cyan
        python training/evaluate.py --task all --dataset kddcup99 --data_path $DATA_PATH --label_col Label --dataset_name kddcup99
        python training/visualize_metrics.py --task intrusion --dataset_name kddcup99
        python training/visualize_metrics.py --task dos --dataset_name kddcup99
        python training/visualize_metrics.py --task port_scan --dataset_name kddcup99

        Write-Host "`n[6/6] Transfer Matrix..." -ForegroundColor Cyan
        python training/compute_transfer.py --dataset_name kddcup99 --unfrozen

    } elseif ($dataset -eq "cicids2017") {
        $DATA_PATH = "..\CICIDS2017"

        if (-not $SkipSSL) {
            Write-Host "`n[1/6] SSL Pretraining (20 epochs)..." -ForegroundColor Cyan
            python training/train_ssl.py --dataset cicids2017 --data_path $DATA_PATH --label_col Label --epochs 20 --dataset_name cicids2017
        }

        Write-Host "`n[2/6] Intrusion Task (20 epochs)..." -ForegroundColor Cyan
        python training/train_task.py --task intrusion --dataset cicids2017 --data_path $DATA_PATH --label_col Label --epochs 20 --dataset_name cicids2017 --unfreeze_encoder --encoder_lr $encoder_lr

        Write-Host "`n[3/6] DoS Task (15 epochs)..." -ForegroundColor Cyan
        python training/train_task.py --task dos --dataset cicids2017 --data_path $DATA_PATH --label_col AttackCategory --epochs 15 --dataset_name cicids2017 --unfreeze_encoder --encoder_lr $encoder_lr

        Write-Host "`n[4/6] Port Scan Task (15 epochs)..." -ForegroundColor Cyan
        python training/train_task.py --task port_scan --dataset cicids2017 --data_path $DATA_PATH --label_col AttackCategory --epochs 15 --dataset_name cicids2017 --unfreeze_encoder --encoder_lr $encoder_lr

        Write-Host "`n[5/6] Evaluation..." -ForegroundColor Cyan
        python training/evaluate.py --task all --dataset cicids2017 --data_path $DATA_PATH --label_col Label --dataset_name cicids2017
        python training/visualize_metrics.py --task intrusion --dataset_name cicids2017
        python training/visualize_metrics.py --task dos --dataset_name cicids2017
        python training/visualize_metrics.py --task port_scan --dataset_name cicids2017

        Write-Host "`n[6/6] Transfer Matrix..." -ForegroundColor Cyan
        python training/compute_transfer.py --dataset_name cicids2017 --unfrozen
    }

    Write-Host "`n$dataset completed!" -ForegroundColor Green
}

Write-Host "`nALL DATASETS RETRAINED SUCCESSFULLY!" -ForegroundColor Green
