param (
    [int]$ssl_epochs = 15,
    [int]$task_epochs = 20,
    [int]$gpm_epochs = 10,
    [string[]]$datasets = @("unsw", "kddcup99", "cicids2017")
)

$ErrorActionPreference = "Stop"

# Ensure we are in the ids-system directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($null -ne $scriptDir -and $scriptDir -ne "") {
    Set-Location $scriptDir
}

# Check if venv is active, if not activate it
if ($null -eq $env:VIRTUAL_ENV) {
    if (Test-Path ".\.venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Cyan
        . .\.venv\Scripts\Activate.ps1
    } else {
        Write-Warning "No virtual environment found at .\.venv. Using default system python."
    }
}

foreach ($dataset in $datasets) {
    Write-Host "`n==========================================" -ForegroundColor Green
    Write-Host "STARTING BENCHMARK FOR DATASET: $dataset" -ForegroundColor Green
    Write-Host "==========================================`n" -ForegroundColor Green

    if ($dataset -eq "unsw") {
        # 1. SSL Pretraining
        if (Test-Path "checkpoints/unsw/encoder_frozen.keras") {
            Write-Host "[1/4] Frozen encoder already exists. Skipping SSL Pretraining." -ForegroundColor Green
        } else {
            Write-Host "[1/4] Running SSL Pretraining..." -ForegroundColor Yellow
            python training/train_ssl.py --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" --label_col "label" --epochs $ssl_epochs --dataset_name unsw
        }

        # 2. Train, Evaluate & Visualize Intrusion
        Write-Host "[2/4] Running Intrusion Task..." -ForegroundColor Yellow
        python training/train_task.py --task intrusion --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" --label_col "label" --epochs $task_epochs --dataset_name unsw
        python training/evaluate.py --task intrusion --test_csv "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv" --label_col "label" --dataset_name unsw
        python training/visualize_metrics.py --task intrusion --dataset_name unsw

        # 3. Train, Evaluate & Visualize DoS
        Write-Host "[3/4] Running DoS Task (Continual GPM)..." -ForegroundColor Yellow
        python training/train_task.py --task dos --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" --label_col "attack_cat" --epochs $gpm_epochs --dataset_name unsw
        python training/evaluate.py --task dos --test_csv "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv" --label_col "attack_cat" --dataset_name unsw
        python training/visualize_metrics.py --task dos --dataset_name unsw

        # 4. Train, Evaluate & Visualize Port Scan
        Write-Host "[4/4] Running Port Scan Task (Continual GPM)..." -ForegroundColor Yellow
        python training/train_task.py --task port_scan --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" --label_col "attack_cat" --epochs $gpm_epochs --dataset_name unsw
        python training/evaluate.py --task port_scan --test_csv "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv" --label_col "attack_cat" --dataset_name unsw
        python training/visualize_metrics.py --task port_scan --dataset_name unsw

    } elseif ($dataset -eq "kddcup99") {
        # 1. SSL Pretraining
        if (Test-Path "checkpoints/kddcup99/encoder_frozen.keras") {
            Write-Host "[1/4] Frozen encoder already exists. Skipping SSL Pretraining." -ForegroundColor Green
        } else {
            Write-Host "[1/4] Running SSL Pretraining..." -ForegroundColor Yellow
            python training/train_ssl.py --dataset kddcup99 --data_path "../KDDCUP99" --label_col "Label" --epochs $ssl_epochs --dataset_name kddcup99
        }

        # 2. Train, Evaluate & Visualize Intrusion
        Write-Host "[2/4] Running Intrusion Task..." -ForegroundColor Yellow
        python training/train_task.py --task intrusion --dataset kddcup99 --data_path "../KDDCUP99" --label_col "Label" --epochs $task_epochs --dataset_name kddcup99
        python training/evaluate.py --task intrusion --dataset kddcup99 --data_path "../KDDCUP99" --label_col "Label" --dataset_name kddcup99
        python training/visualize_metrics.py --task intrusion --dataset_name kddcup99

        # 3. Train, Evaluate & Visualize DoS
        Write-Host "[3/4] Running DoS Task (Continual GPM)..." -ForegroundColor Yellow
        python training/train_task.py --task dos --dataset kddcup99 --data_path "../KDDCUP99" --label_col "AttackCategory" --epochs $gpm_epochs --dataset_name kddcup99
        python training/evaluate.py --task dos --dataset kddcup99 --data_path "../KDDCUP99" --label_col "AttackCategory" --dataset_name kddcup99
        python training/visualize_metrics.py --task dos --dataset_name kddcup99

        # 4. Train, Evaluate & Visualize Port Scan
        Write-Host "[4/4] Running Port Scan Task (Continual GPM)..." -ForegroundColor Yellow
        python training/train_task.py --task port_scan --dataset kddcup99 --data_path "../KDDCUP99" --label_col "AttackCategory" --epochs $gpm_epochs --dataset_name kddcup99
        python training/evaluate.py --task port_scan --dataset kddcup99 --data_path "../KDDCUP99" --label_col "AttackCategory" --dataset_name kddcup99
        python training/visualize_metrics.py --task port_scan --dataset_name kddcup99

    } elseif ($dataset -eq "cicids2017") {
        # 1. SSL Pretraining
        if (Test-Path "checkpoints/cicids2017/encoder_frozen.keras") {
            Write-Host "[1/4] Frozen encoder already exists. Skipping SSL Pretraining." -ForegroundColor Green
        } else {
            Write-Host "[1/4] Running SSL Pretraining..." -ForegroundColor Yellow
            python training/train_ssl.py --dataset cicids2017 --data_path "../CICIDS2017" --label_col "Label" --epochs $ssl_epochs --dataset_name cicids2017
        }

        # 2. Train, Evaluate & Visualize Intrusion
        Write-Host "[2/4] Running Intrusion Task..." -ForegroundColor Yellow
        python training/train_task.py --task intrusion --dataset cicids2017 --data_path "../CICIDS2017" --label_col "Label" --epochs $task_epochs --dataset_name cicids2017
        python training/evaluate.py --task intrusion --dataset cicids2017 --data_path "../CICIDS2017" --label_col "Label" --dataset_name cicids2017
        python training/visualize_metrics.py --task intrusion --dataset_name cicids2017

        # 3. Train, Evaluate & Visualize DoS
        Write-Host "[3/4] Running DoS Task (Continual GPM)..." -ForegroundColor Yellow
        python training/train_task.py --task dos --dataset cicids2017 --data_path "../CICIDS2017" --label_col "AttackCategory" --epochs $gpm_epochs --dataset_name cicids2017
        python training/evaluate.py --task dos --dataset cicids2017 --data_path "../CICIDS2017" --label_col "AttackCategory" --dataset_name cicids2017
        python training/visualize_metrics.py --task dos --dataset_name cicids2017

        # 4. Train, Evaluate & Visualize Port Scan
        Write-Host "[4/4] Running Port Scan Task (Continual GPM)..." -ForegroundColor Yellow
        python training/train_task.py --task port_scan --dataset cicids2017 --data_path "../CICIDS2017" --label_col "AttackCategory" --epochs $gpm_epochs --dataset_name cicids2017
        python training/evaluate.py --task port_scan --dataset cicids2017 --data_path "../CICIDS2017" --label_col "AttackCategory" --dataset_name cicids2017
        python training/visualize_metrics.py --task port_scan --dataset_name cicids2017
    }
}
Write-Host "`nAll datasets and task heads processed successfully!" -ForegroundColor Green
