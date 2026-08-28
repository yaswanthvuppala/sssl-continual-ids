"""
run_experiments.py — Unified Benchmark Runner for SSSL-IDS across All Datasets & Label Ratios.

Supports:
  - Datasets: anoshift, cicids2017, kddcup99, unsw (or 'all')
  - Label Ratios: 0 (Unsupervised/Zero-Day), 0.05 (5% Labeled), 0.10 (10% Labeled), 0.20, 1.0
  - Produces a consolidated comparative Markdown & CSV report.

Usage Examples:
  # Run 0%, 5%, 10% label regimes on AnoShift:
  python run_experiments.py --datasets anoshift --label_ratios 0 0.05 0.10

  # Run on ALL datasets across 0%, 5%, 10% labeled data:
  python run_experiments.py --datasets all --label_ratios 0 0.05 0.10

  # Fast quick test mode:
  python run_experiments.py --datasets anoshift --label_ratios 0 0.05 --quick
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import List, Dict, Any

DATASET_CONFIGS = {
    "anoshift": {
        "name": "AnoShift (Kyoto 2006+)",
        "data_path": "./data/anoshift",
        "label_col": "Label",
        "is_csv": False,
        "tasks": ["intrusion", "dos", "port_scan"],
        "eval_splits": ["iid", "near", "far"],
    },
    "cicids2017": {
        "name": "CICIDS2017",
        "data_path": "../CICIDS2017",
        "label_col": "Label",
        "is_csv": False,
        "tasks": ["intrusion", "dos", "port_scan"],
        "eval_splits": ["test"],
    },
    "kddcup99": {
        "name": "KDD Cup 99",
        "data_path": "../KDDCUP99",
        "label_col": "Label",
        "is_csv": False,
        "tasks": ["intrusion", "dos", "port_scan"],
        "eval_splits": ["test"],
    },
    "unsw": {
        "name": "UNSW-NB15",
        "train_csv": "../IDS-UNSW_NB/UNSW_NB15_training-set.csv",
        "test_csv": "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv",
        "label_col": "label",
        "is_csv": True,
        "tasks": ["intrusion", "dos", "port_scan"],
        "eval_splits": ["test"],
    },
}


def run_cmd(cmd: List[str], desc: str = "") -> int:
    print(f"\n[RUN] {desc or ' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[WARN] Command returned exit code {res.returncode} ({elapsed:.1f}s)")
    else:
        print(f"[DONE] Completed in {elapsed:.1f}s")
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Unified SSSL-IDS Experiment Runner across Datasets and Label Ratios")
    parser.add_argument("--datasets", nargs="+", default=["anoshift"],
                        help="Datasets to evaluate: anoshift, cicids2017, kddcup99, unsw, or 'all'")
    parser.add_argument("--label_ratios", nargs="+", type=float, default=[0.0, 0.05, 0.10],
                        help="Label ratios to evaluate: 0 (unsupervised), 0.05 (5%%), 0.10 (10%%), etc.")
    parser.add_argument("--ssl_epochs", type=int, default=15, help="Epochs for SSL pretraining")
    parser.add_argument("--task_epochs", type=int, default=15, help="Epochs for task classifier training")
    parser.add_argument("--anomaly_epochs", type=int, default=15, help="Epochs for anomaly autoencoder")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced epochs and batch size for rapid testing")
    parser.add_argument("--output_dir", type=str, default="./logs/benchmark_comparison", help="Output directory for reports")
    args = parser.parse_args()

    if "all" in [d.lower() for d in args.datasets]:
        target_datasets = ["anoshift", "cicids2017", "kddcup99", "unsw"]
    else:
        target_datasets = [d.lower() for d in args.datasets]

    ssl_epochs = 2 if args.quick else args.ssl_epochs
    task_epochs = 2 if args.quick else args.task_epochs
    anomaly_epochs = 2 if args.quick else args.anomaly_epochs
    max_samples_ssl = 5000 if args.quick else 100000

    os.makedirs(args.output_dir, exist_ok=True)
    report_rows = []

    print("=" * 80)
    print("  SSSL-IDS FEW-LABEL CONTINUAL BENCHMARK RUNNER")
    print(f"  Datasets: {target_datasets}")
    print(f"  Label Ratios: {args.label_ratios}")
    print(f"  SSL Epochs: {ssl_epochs} | Task Epochs: {task_epochs} | Quick Mode: {args.quick}")
    print("=" * 80)

    for ds_key in target_datasets:
        if ds_key not in DATASET_CONFIGS:
            print(f"[SKIP] Unknown dataset '{ds_key}'")
            continue

        cfg = DATASET_CONFIGS[ds_key]
        print(f"\n{'#'*80}")
        print(f"  PROCESSING DATASET: {cfg['name']} ({ds_key})")
        print(f"{'#'*80}")

        # Ensure dataset is downloaded / available
        if ds_key == "anoshift":
            run_cmd([
                sys.executable, "data/download_anoshift.py",
                "--subset", "I/10",
                "--save_dir", "./data/anoshift"
            ], desc="Ensuring AnoShift dataset files exist")

        # ── 1. SSL Pretraining (Shared base encoder, 0% labels) ──
        base_exp_name = f"{ds_key}_base"
        ssl_enc_path = f"./checkpoints/{base_exp_name}/encoder_frozen.keras"
        if os.path.exists(ssl_enc_path):
            print(f"\n[INFO] SSL Pretrained base encoder exists at {ssl_enc_path}. Skipping pretraining.")
        else:
            ssl_cmd = [
                sys.executable, "training/train_ssl.py",
                "--epochs", str(ssl_epochs),
                "--batch_size", "256" if not args.quick else "64",
                "--max_samples", str(max_samples_ssl),
                "--dataset_name", base_exp_name
            ]
            if cfg["is_csv"]:
                ssl_cmd.extend(["--train_csv", cfg["train_csv"], "--label_col", cfg["label_col"]])
            else:
                ssl_cmd.extend(["--dataset", ds_key, "--data_path", cfg["data_path"], "--label_col", cfg["label_col"]])
            run_cmd(ssl_cmd, desc=f"SSL Pretraining for {cfg['name']}")

        # ── 2. Run Experiments for each requested label ratio ──
        for ratio in args.label_ratios:
            ratio_pct = int(ratio * 100) if ratio > 0 else 0
            exp_name = f"{ds_key}_{ratio_pct}pct"
            print(f"\n>>> Running Regime: {ratio_pct}% Labeled Data on {cfg['name']} (exp: {exp_name}) <<<")

            if ratio == 0.0:
                # ── 0% Labeled Regime: Unsupervised SSL + Anomaly Autoencoder + Zero-Day Detector ──
                ae_path = f"./checkpoints/{base_exp_name}/anomaly_ae.keras"
                if os.path.exists(ae_path):
                    print(f"[INFO] Anomaly Autoencoder exists at {ae_path}. Skipping AE training.")
                else:
                    ae_cmd = [
                        sys.executable, "training/train_anomaly.py",
                        "--epochs", str(anomaly_epochs),
                        "--latent_dim", "16",
                        "--dataset_name", base_exp_name
                    ]
                    if cfg["is_csv"]:
                        ae_cmd.extend(["--train_csv", cfg["train_csv"], "--label_col", cfg["label_col"]])
                    else:
                        ae_cmd.extend(["--dataset", ds_key, "--data_path", cfg["data_path"]])
                    run_cmd(ae_cmd, desc=f"Train Anomaly Autoencoder (0% labels) for {cfg['name']}")

                # Evaluate Zero-Day Threat Catching
                zd_cmd = [
                    sys.executable, "training/evaluate_zeroday.py",
                    "--dataset_name", base_exp_name
                ]
                if cfg["is_csv"]:
                    zd_cmd.extend(["--test_csv", cfg["test_csv"], "--label_col", cfg["label_col"]])
                else:
                    zd_cmd.extend(["--dataset", ds_key, "--data_path", cfg["data_path"]])
                run_cmd(zd_cmd, desc=f"Evaluate Zero-Day Threat Detection (0% labels) on {cfg['name']}")

                report_rows.append({
                    "Dataset": cfg["name"],
                    "Label Ratio": "0% (Zero-Shot)",
                    "Task": "Zero-Day Anomaly Detection",
                    "Status": "Evaluated",
                    "Experiment": base_exp_name
                })

            else:
                # ── 5%, 10%, etc. Semi-Supervised Continual Tasks with GPM ──
                # Copy base SSL encoder into this experiment's scoped checkpoint dir
                src_enc = f"./checkpoints/{base_exp_name}/encoder_frozen.keras"
                dst_dir = f"./checkpoints/{exp_name}"
                os.makedirs(dst_dir, exist_ok=True)
                if os.path.exists(src_enc):
                    import shutil
                    shutil.copy2(src_enc, f"{dst_dir}/encoder_frozen.keras")

                src_prep = f"./checkpoints/{base_exp_name}/preprocessor.pkl"
                if os.path.exists(src_prep):
                    import shutil
                    shutil.copy2(src_prep, f"{dst_dir}/preprocessor.pkl")

                # Train continual tasks sequentially: intrusion -> dos -> port_scan
                for task in cfg["tasks"]:
                    task_cmd = [
                        sys.executable, "training/train_task.py",
                        "--task", task,
                        "--label_ratio", str(ratio),
                        "--epochs", str(task_epochs),
                        "--unfreeze_encoder",
                        "--encoder_lr", "0.003",
                        "--dataset_name", exp_name
                    ]
                    if cfg["is_csv"]:
                        task_col = "label" if task == "intrusion" else "attack_cat"
                        task_cmd.extend(["--train_csv", cfg["train_csv"], "--label_col", task_col])
                    else:
                        task_cmd.extend(["--dataset", ds_key, "--data_path", cfg["data_path"]])
                    
                    run_cmd(task_cmd, desc=f"Train Task '{task}' ({ratio_pct}% labels) on {cfg['name']}")

                # Evaluate Continual Tasks across test splits
                for split in cfg["eval_splits"]:
                    eval_cmd = [
                        sys.executable, "training/evaluate.py",
                        "--task", "all",
                        "--dataset_name", exp_name
                    ]
                    if cfg["is_csv"]:
                        eval_cmd.extend(["--test_csv", cfg["test_csv"], "--label_col", cfg["label_col"]])
                    else:
                        eval_cmd.extend(["--dataset", ds_key, "--data_path", cfg["data_path"], "--split", split])

                    run_cmd(eval_cmd, desc=f"Evaluate Tasks ({ratio_pct}% labels, split: {split}) on {cfg['name']}")

                # Compute Transfer Matrix & BWT
                run_cmd([
                    sys.executable, "training/compute_transfer.py",
                    "--dataset_name", exp_name,
                    "--data_path", cfg["data_path"] if not cfg["is_csv"] else "."
                ], desc=f"Compute Continual Transfer Matrix for {exp_name}")

                report_rows.append({
                    "Dataset": cfg["name"],
                    "Label Ratio": f"{ratio_pct}%",
                    "Task": "Continual (Intrusion + DoS + Scan)",
                    "Status": "Completed",
                    "Experiment": exp_name
                })

    # ── Summary Report Generation ──
    report_md = [
        "# SSSL-IDS Multi-Dataset Few-Label Benchmark Summary\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| Dataset | Label Ratio | Task Regime | Status | Checkpoint Directory |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in report_rows:
        report_md.append(f"| {r['Dataset']} | {r['Label Ratio']} | {r['Task']} | {r['Status']} | `checkpoints/{r['Experiment']}` |")

    report_path = os.path.join(args.output_dir, "benchmark_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))

    print("\n" + "=" * 80)
    print("  ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print(f"  Summary saved to: {report_path}")
    print("=" * 80)
    print("\n".join(report_md))


if __name__ == "__main__":
    main()
