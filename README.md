# SSSL-Based Continual Intrusion Detection System

A research-grade, terminal-only IDS framework combining **Self-Supervised Learning (SimCLR)**, **Semi-Supervised Learning (FixMatch + Focal Loss)**, and **Continual Learning (GPM)** for network intrusion detection on the UNSW-NB15 dataset.

---

## Architecture

```
Stage 1 — SSL Pretraining (SimCLR)
    Unlabeled Traffic → Flow Encoder → Frozen Embeddings

Stage 2 — Task-Specific Heads (FixMatch + Focal Loss + GPM)
    Frozen Encoder → Classifier Head (per attack type) → IDS Alert
    GPM projects gradients to null-space of past tasks → No Forgetting

Anomaly Detection:
    Frozen Encoder → Autoencoder → Zero-Day Score
```

---

## Step-by-Step: How to Run

### Prerequisites

```powershell
# Navigate to the project
cd C:\Users\vuppa\Desktop\SSSL_Based_IDS\ids-system

# Activate the virtual environment
.\.venv\Scripts\activate

# Install dependencies (first time only)
pip install -r requirements.txt
```

---

### Option A — Full Pipeline (Recommended, One Command)

Trains SSL encoder + intrusion head + evaluates + generates all plots automatically.

```powershell
python main.py --mode unsw --ssl_epochs 15 --task_epochs 20
```

This runs in order:
1. SSL pretraining on `UNSW_NB15_training-set.csv`
2. Intrusion task training (FixMatch + Focal Loss + class weights)
3. Evaluation on `UNSW_NB15_testing-set.csv`
4. All visualization plots saved to `./logs/plots/`

---

### Option B — Step by Step (Full Continual Learning)

Run each stage manually to train all 3 task heads with GPM anti-forgetting.

#### Step 1 — SSL Pretraining
```powershell
python main.py --mode ssl `
  --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" `
  --label_col "label" `
  --epochs 15
```
> Saves frozen encoder to `./checkpoints/encoder_frozen.keras`

---

#### Step 2 — Train Intrusion Head (Binary: Normal vs Attack)
```powershell
python main.py --mode task --task intrusion `
  --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" `
  --label_col "label" `
  --epochs 20
```
> Saves checkpoint to `./checkpoints/intrusion/`  
> Saves training history to `./logs/task_intrusion/training_history.json`

---

#### Step 3 — Train DoS Head (Continual, GPM enabled)
```powershell
python main.py --mode task --task dos `
  --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" `
  --label_col "attack_cat" `
  --epochs 10
```
> GPM captures gradient basis (6 components) and saves to `./checkpoints/gpm/memory_bank.pkl`

---

#### Step 4 — Train Port Scan Head (Continual, GPM enabled)
```powershell
python main.py --mode task --task port_scan `
  --train_csv "../IDS-UNSW_NB/UNSW_NB15_training-set.csv" `
  --label_col "attack_cat" `
  --epochs 10
```
> GPM loads dos basis, protects it, then adds port_scan basis (14 components)  
> Memory bank now has 2 task bases total

---

#### Step 5 — Evaluate
```powershell
python main.py --mode evaluate --task intrusion `
  --test_csv "../IDS-UNSW_NB/UNSW_NB15_testing-set.csv" `
  --label_col "label"
```
> Prints Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC  
> Reports **optimal decision threshold** (found via PR-curve, maximizes F1)  
> Saves confusion matrix to `./logs/eval/cm_intrusion.png`  
> Saves all metrics to `./logs/eval/metrics_intrusion.json`

---

#### Step 6 — Generate Visualization Plots
```powershell
python main.py --mode visualize --task intrusion
```

Generates 4 plots in `./logs/plots/`:

| Plot File | Contents |
|-----------|----------|
| `cl_metrics_dashboard.png` | Training loss curves, pseudo-label mask rate, per-task metrics, per-class recall |
| `evaluation_metrics_intrusion.png` | ROC curve, PR curve, confusion matrix heatmap, per-class bar chart |
| `threshold_analysis_intrusion.png` | Precision / Recall / F1 vs decision threshold sweep |
| `memory_hierarchy.png` | GPM basis dimensionality, cumulative gradient subspace, SVD spectrum |

---

### Option C — Benchmark (Synthetic Data, No CSV needed)

```powershell
python main.py --mode benchmark
```

Runs the full pipeline on auto-generated synthetic data. Useful for verifying the setup works before using real data.

---

## Key Results (UNSW-NB15)

| Mode | Attack Recall | Accuracy | F1 | ROC-AUC |
|------|--------------|----------|----|---------|
| Default threshold (0.5) | 0.66 | 0.77 | 0.77 | 0.978 |
| **Optimal threshold (~0.001)** | **0.99** | **0.93** | **0.93** | **0.978** |

> The optimal threshold is automatically found by the evaluation script using PR-curve analysis.

---

## Preprocessor Files (Important)

Each task uses its own preprocessor because label columns differ:

| Task | Label Column | Preprocessor File |
|------|-------------|-------------------|
| `intrusion` | `label` (binary 0/1) | `checkpoints/preprocessor.pkl` |
| `dos` | `attack_cat` (strings) | `checkpoints/preprocessor_dos.pkl` |
| `port_scan` | `attack_cat` (strings) | `checkpoints/preprocessor_port_scan.pkl` |

---

## Project Structure

```
ids-system/
├── encoder/          # SimCLR encoder + projection head + NT-Xent loss
├── gpm/              # Gradient Projection Memory (SVD-based anti-forgetting)
├── classifiers/      # Task heads + FixMatch trainer (Focal Loss + class weights)
├── anomaly/          # Autoencoder detector
├── data/             # Dataset loading, preprocessing, augmentations, tf.data
├── training/
│   ├── train_ssl.py         # Stage 1: SSL pretraining
│   ├── train_task.py        # Stage 2: Task head training
│   ├── evaluate.py          # Evaluation + threshold search
│   ├── visualize_metrics.py # All visualization plots
│   └── benchmark.py         # End-to-end synthetic benchmark
├── inference/        # Inference engine + predict CLI
├── checkpoints/      # Saved model weights + GPM memory bank
├── logs/
│   ├── eval/         # Confusion matrices + metrics JSON
│   ├── plots/        # All visualization plots
│   └── task_*/       # TensorBoard logs + training history per task
├── main.py           # Master CLI entry point
└── requirements.txt
```

---

## All CLI Modes

| Mode | Command | Description |
|------|---------|-------------|
| `ssl` | `python main.py --mode ssl` | Stage 1: SSL pretraining |
| `task` | `python main.py --mode task --task intrusion` | Train a specific task head |
| `evaluate` | `python main.py --mode evaluate --task intrusion` | Evaluate with optimal threshold |
| `visualize` | `python main.py --mode visualize --task intrusion` | Generate all plots |
| `unsw` | `python main.py --mode unsw` | Full UNSW-NB15 pipeline (1 command) |
| `pipeline` | `python main.py --mode pipeline --train_csv ... --test_csv ...` | Custom CSV pipeline |
| `benchmark` | `python main.py --mode benchmark` | Synthetic end-to-end benchmark |
| `predict` | `python main.py --mode predict` | Run inference |

---

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- NumPy, pandas, scikit-learn, matplotlib, seaborn, tqdm, PyYAML
