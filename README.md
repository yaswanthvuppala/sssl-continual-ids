# SSSL-Based Continual Intrusion Detection System

A research-grade, terminal-only IDS framework combining **Self-Supervised Learning (SimCLR)**, **Semi-Supervised Learning (FixMatch + Focal Loss)**, and **Continual Learning (GPM)** for network intrusion detection on UNSW-NB15, CICIDS2017, and KDD Cup 99.

See [`DATASET_SUPPORT_CICIDS2017_KDDCUP99.md`](DATASET_SUPPORT_CICIDS2017_KDDCUP99.md)
for the exact dataset files, split policy, label mapping, and commands.

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

### Inference Engine & Decision Logic

To yield highly calibrated alerts across multi-task heads, the inference pipeline (`IDSInferenceEngine`) employs:
- **Temperature Scaling (Calibration)**: During validation/evaluation, logit outputs for each task are calibrated using a fitted temperature parameter $T > 0$ via a `TemperatureScaler`. These parameters are stored under `logs/{dataset_name}/eval/temperature_{task_name}.json`.
- **Per-Head Thresholding**: Each task classifier head uses its own optimal classification threshold (loaded from `metrics_{task_name}.json`) instead of a single global threshold.
- **Margin-Based Decision Routing**: Logits are scaled by task temperature and converted to probabilities. The engine calculates the margins:
  $$\text{Margin} = P(\text{attack}) - \text{Threshold}$$
  The system triggers the alert of the classifier head with the **highest positive margin**.
- **Zero-Day Detector Fallback**: If no head yields a positive margin, the encoder embedding is sent to the Anomaly Autoencoder. A reconstruction error exceeding the anomaly threshold triggers a `zero-day / unknown` alert; otherwise, the flow is marked benign.

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

### Understanding the Tasks & Order of Execution

To ensure the model learns correctly without forgetting, the tasks **must** be executed in the following order:
1. **SSL Pretraining**: Trains the base feature representation.
2. **Intrusion Task**: Learns general intrusion detection using pre-trained features (normal vs attack).
3. **DoS Task**: Learns DoS signatures while using GPM (Gradient Projection Memory) to protect the general intrusion representation.
4. **Port Scan Task**: Learns port scans while using GPM to protect *both* intrusion and DoS representations.

---

### Option A — Run All Datasets and Tasks at Once (Recommended)

You can run the orchestrator script `run_all.ps1` to automatically process SSL pretraining, followed by all three tasks (`intrusion`, `dos`, `port_scan`), sequentially across all three datasets (`unsw`, `kddcup99`, and `cicids2017`). All evaluation reports and visualization plots are generated and saved automatically.

```powershell
.\run_all.ps1 -ssl_epochs 15 -task_epochs 20 -gpm_epochs 10
```

To run only a specific dataset (e.g., just `unsw`):
```powershell
.\run_all.ps1 -ssl_epochs 15 -task_epochs 20 -gpm_epochs 10 -datasets unsw
```

---

### Option B — Run Individually Using the CLI

If you prefer to run specific datasets or tasks manually, use `main.py`.

#### 1. UNSW-NB15
```powershell
# Binary Intrusion Detection (normal vs attack)
python main.py --mode unsw --task intrusion --ssl_epochs 15 --task_epochs 20

# DoS Detection (requires attack_cat mapping)
python main.py --mode unsw --task dos --label_col "attack_cat" --ssl_epochs 15 --task_epochs 20

# Port Scan/Probe Detection (requires attack_cat mapping)
python main.py --mode unsw --task port_scan --label_col "attack_cat" --ssl_epochs 15 --task_epochs 20
```

#### 2. KDD Cup 99
```powershell
# Binary Intrusion Detection
python main.py --mode kddcup99 --task intrusion --ssl_epochs 15 --task_epochs 20

# DoS Detection
python main.py --mode kddcup99 --task dos --ssl_epochs 15 --task_epochs 20

# Probe Detection (Port Scan)
python main.py --mode kddcup99 --task port_scan --ssl_epochs 15 --task_epochs 20
```

#### 3. CICIDS2017
```powershell
# Binary Intrusion Detection
python main.py --mode cicids2017 --task intrusion --ssl_epochs 15 --task_epochs 20

# DoS Detection
python main.py --mode cicids2017 --task dos --ssl_epochs 15 --task_epochs 20

# Port Scan Detection
python main.py --mode cicids2017 --task port_scan --ssl_epochs 15 --task_epochs 20
```

---

### Outputs Location
All outputs are dataset-scoped to prevent path collisions:
- **Plots**: `logs/{dataset_name}/plots/`
- **Checkpoints**: `checkpoints/{dataset_name}/`
- **Training Logs & Evaluation Logs**: `logs/{dataset_name}/` (contains training summaries, `eval/metrics_{task_name}.json` with optimal thresholds, and `eval/temperature_{task_name}.json` for logit temperature scaling)

---

## Preprocessor Files (Important)

Each dataset has its own preprocessor because feature dimensions and columns differ between datasets:

| Dataset | Preprocessor File |
|---------|-------------------|
| UNSW-NB15 | `checkpoints/unsw/preprocessor.pkl` |
| KDD Cup 99 | `checkpoints/kddcup99/preprocessor.pkl` |
| CICIDS2017 | `checkpoints/cicids2017/preprocessor.pkl` |

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
| `cicids2017` | `python main.py --mode cicids2017` | Full CICIDS2017 pipeline |
| `kddcup99` | `python main.py --mode kddcup99` | Full KDD Cup 99 pipeline |
| `pipeline` | `python main.py --mode pipeline --train_csv ... --test_csv ...` | Custom CSV pipeline |
| `benchmark` | `python main.py --mode benchmark` | Synthetic end-to-end benchmark |
| `predict` | `python main.py --mode predict` | Run inference |

---

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- NumPy, pandas, scikit-learn, matplotlib, seaborn, tqdm, PyYAML
