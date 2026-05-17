# SSSL-Based Continual Intrusion Detection System

A research-grade, terminal-only IDS framework combining **Self-Supervised Learning**, **Semi-Supervised Learning (FixMatch)**, and **Continual Learning (GPM)** for network intrusion detection.

## Architecture

```
Stage 1: SSL Pretraining (SimCLR)
    Unlabeled Traffic → Flow Encoder → Frozen Embeddings

Stage 2: Task-Specific Heads (FixMatch + GPM)
    Frozen Encoder → Classifier Head (per attack type) → IDS Alert
    GPM projects gradients to null-space of past tasks → No Forgetting

Anomaly Detection:
    Frozen Encoder → Autoencoder / Isolation Forest → Zero-Day Score
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Full end-to-end benchmark (recommended first run)
python main.py --mode benchmark

# Train and test with CSV paths supplied from the terminal
python main.py --mode pipeline --train_csv path/to/training.csv --test_csv path/to/testing.csv --label_col Label --ssl_epochs 5 --task_epochs 5

# Train on UNSW-NB15 training-set.csv and evaluate on testing-set.csv
python main.py --mode unsw --ssl_epochs 5 --task_epochs 5

# Or step by step:
python main.py --mode ssl --epochs 10          # Stage 1
python main.py --mode task --task dos           # Stage 2: DoS head
python main.py --mode task --task port_scan     # Stage 2: Port Scan head
python main.py --mode evaluate                  # Metrics
python main.py --mode predict                   # Inference
```

For a manual UNSW-NB15 run:

```bash
python main.py --mode ssl --train_csv ../IDS-UNSW_NB/UNSW_NB15_training-set.csv --label_col label --epochs 5
python main.py --mode task --task intrusion --train_csv ../IDS-UNSW_NB/UNSW_NB15_training-set.csv --label_col label --epochs 5
python main.py --mode evaluate --task intrusion --test_csv ../IDS-UNSW_NB/UNSW_NB15_testing-set.csv --label_col label
```

PowerShell example with explicit dataset paths:

```powershell
python main.py --mode pipeline `
  --train_csv ../IDS-UNSW_NB/UNSW_NB15_training-set.csv `
  --test_csv ../IDS-UNSW_NB/UNSW_NB15_testing-set.csv `
  --label_col label `
  --ssl_epochs 5 `
  --task_epochs 5
```

## Project Structure

```
ids-system/
├── encoder/                 # SSL encoder + projection head + NT-Xent loss
├── gpm/                     # Gradient Projection Memory (SVD-based)
├── classifiers/             # Task-specific heads + FixMatch trainer
├── anomaly/                 # Autoencoder & Isolation Forest detectors
├── data/                    # Dataset loading, preprocessing, augmentations, tf.data
├── training/                # train_ssl.py, train_task.py, evaluate.py, benchmark.py
├── inference/               # Inference engine + CLI predict script
├── configs/                 # YAML configurations
├── checkpoints/             # Saved model weights
├── logs/                    # TensorBoard logs & evaluation outputs
├── tests/                   # Unit tests
├── main.py                  # Master CLI entry point
└── requirements.txt
```

## Modules

| Module | Purpose |
|--------|---------|
| **SSL Encoder** | SimCLR-based self-supervised pretraining on unlabeled flows |
| **GPM** | Gradient Projection Memory for catastrophic forgetting prevention |
| **FixMatch** | Semi-supervised training with pseudo-labels (100–500 labeled samples) |
| **Anomaly Detector** | Autoencoder + Isolation Forest for zero-day detection |
| **Inference Engine** | Parallel head scoring + anomaly detection → structured alerts |

## Terminal Commands

| Command | Description |
|---------|-------------|
| `python training/train_ssl.py` | SSL pretraining |
| `python training/train_task.py --task dos` | Train DoS classifier |
| `python training/train_task.py --task port_scan` | Train Port Scan classifier |
| `python training/evaluate.py` | Evaluate all heads |
| `python inference/predict.py` | Run inference |
| `python training/benchmark.py` | Full end-to-end benchmark |

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrices (saved as PNG)
- Forgetting Matrix (continual learning)

## Continual Learning Flow

1. Train Task 1 (DoS) → Capture gradient basis via GPM
2. Train Task 2 (Port Scan) → GPM projects gradients to null-space → Task 1 protected
3. Train Task N+1 → All previous tasks remain stable

## Dataset Support

- CICIDS2017, CIC-IDS2018, UNSW-NB15, TON_IoT
- Custom CSV datasets
- Built-in synthetic data generator for testing

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- NumPy, pandas, scikit-learn, matplotlib, tqdm, PyYAML
