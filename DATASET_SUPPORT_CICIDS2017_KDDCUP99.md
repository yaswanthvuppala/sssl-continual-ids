# CICIDS2017 and KDD Cup 99 Dataset Support

## 1. Goal

The IDS pipeline now accepts:

- CICIDS2017 machine-learning CSV files
- KDD Cup 99 raw data files without header rows
- Existing UNSW-NB15 and generic CSV inputs

The model architecture was not changed. The encoder, projection head, task
heads, FixMatch trainer, and Gradient Projection Memory retain their original
layer definitions and training behavior.

## 2. What Was Changed

Only the data and command-line integration layers were extended:

- `ids-system/data/dataset_loader.py`
  - Parses KDD Cup 99's 41 headerless feature columns.
  - Loads all CICIDS2017 CSV files from a directory.
  - Creates deterministic CICIDS2017 train and test splits.
  - Adds consistent binary and attack-family labels.
- `ids-system/data/preprocessing.py`
  - Prevents all generated label metadata from entering the feature tensor.
- `ids-system/training/train_ssl.py`
- `ids-system/training/train_task.py`
- `ids-system/training/evaluate.py`
  - Accept `--dataset` and `--data_path`.
- `ids-system/main.py`
  - Adds one-command `kddcup99` and `cicids2017` modes.
- `ids-system/tests/test_dataset_loader.py`
  - Tests parsing, splitting, label conversion, and label-leakage prevention.

No files under `ids-system/encoder/`, `ids-system/classifiers/`, or
`ids-system/gpm/` were structurally modified.

## 3. KDD Cup 99: Exact Train and Test Choice

### Selected training data

```text
KDDCUP99/kddcup.data_10_percent/kddcup.data_10_percent
```

- Purpose: training
- Rows in the local file: **494,021**
- Format: 41 features followed by a labeled attack name
- Reason: this is the official labeled 10% training subset and is much more
  practical for the current in-memory training pipeline than the full file.

### Selected test data

```text
KDDCUP99/corrected/corrected
```

- Purpose: final evaluation
- Rows in the local file: **311,029**
- Format: 41 features followed by corrected test labels
- Reason: UCI explicitly identifies `corrected.gz` as test data with corrected
  labels.

### Files not selected

| File | Why it is not used by the preset |
|---|---|
| `kddcup.data/kddcup.data` | Full labeled training data with 4,898,431 rows; valid but much heavier than the chosen 10% training subset. |
| `kddcup.testdata.unlabeled*` | No labels, so accuracy, recall, F1, ROC-AUC, and confusion matrices cannot be calculated. |
| `kddcup.newtestdata_10_percent_unlabeled*` | Also unlabeled and therefore unsuitable for the evaluation script. |
| `kddcup.data.corrected` | A corrected training-data copy, not the official corrected labeled test file selected above. |

Therefore, the KDD preset is:

```text
TRAIN = kddcup.data_10_percent
TEST  = corrected
```

This is not a new random split. It follows the labeled training/test roles
published with the KDD Cup 99 archive.

## 4. CICIDS2017: Train and Test Choice

CICIDS2017 provides labeled CSV files for different days and attacks, but it
does not publish one official machine-learning train/test partition.

The implemented preset:

1. Finds every `.csv` file recursively under `CICIDS2017/`.
2. Cleans whitespace from column names.
3. Processes each CSV separately.
4. Performs an **80% training / 20% test** split within that CSV.
5. Uses stratification on the original attack label.
6. Uses `random_state=42`, so every training and evaluation process recreates
   the same split.
7. Concatenates the selected portions from all CSV files.

Splitting each file separately ensures that every CICIDS2017 day contributes to
both sets and preserves rare attack labels better than one unstratified split.

This is a reproducible flow-level benchmark split, not an official UNB split.
For a strict time-based or cross-day experiment, use explicit prepared train
and test CSV files with `--mode pipeline`.

## 5. Standardized Labels

The loader adds three label columns before preprocessing:

| Column | Meaning | Example |
|---|---|---|
| `Label` | Binary intrusion label | `normal`, `attack` |
| `AttackLabel` | Original normalized dataset label | `smurf`, `DDoS`, `PortScan` |
| `AttackCategory` | Common attack family | `dos`, `probe`, `r2l`, `u2r` |

The preprocessor drops all three label columns from model inputs except for the
one currently selected as the target. This prevents label leakage.

Task selection:

| Task | Label column used by preset | Positive class |
|---|---|---|
| `intrusion` | `Label` | Any attack |
| `dos` | `AttackCategory` | `dos` |
| `port_scan` | `AttackCategory` | `probe` |

For KDD Cup 99, the `port_scan` head treats the official `probe` family as the
positive class because that family contains attacks such as `portsweep`,
`ipsweep`, `nmap`, and `satan`.

## 6. Commands

Run commands from the model directory:

```powershell
cd C:\Users\vuppa\Desktop\SSSL_Based_IDS\ids-system
```

### KDD Cup 99 binary intrusion pipeline

```powershell
python main.py --mode kddcup99 --ssl_epochs 15 --task_epochs 20
```

The default data directory is `../KDDCUP99`.

### CICIDS2017 binary intrusion pipeline

```powershell
python main.py --mode cicids2017 --ssl_epochs 15 --task_epochs 20
```

The default data directory is `../CICIDS2017`.

### DoS task

```powershell
python main.py --mode kddcup99 --task dos --ssl_epochs 15 --task_epochs 20
python main.py --mode cicids2017 --task dos --ssl_epochs 15 --task_epochs 20
```

### Probe or port-scan task

```powershell
python main.py --mode kddcup99 --task port_scan --ssl_epochs 15 --task_epochs 20
python main.py --mode cicids2017 --task port_scan --ssl_epochs 15 --task_epochs 20
```

### Custom dataset location

```powershell
python main.py --mode kddcup99 `
  --data_path "D:\datasets\KDDCUP99" `
  --ssl_epochs 15 `
  --task_epochs 20
```

Stage-specific commands also work:

```powershell
python main.py --mode ssl `
  --dataset kddcup99 `
  --data_path "../KDDCUP99" `
  --label_col "Label" `
  --epochs 15
```

## 7. Important Checkpoint Rule

KDD Cup 99, CICIDS2017, and UNSW-NB15 produce different input dimensions after
categorical encoding. Always run SSL pretraining for the selected dataset
before training its task head. Do not combine an encoder or preprocessor from
one dataset with features from another dataset.

The one-command dataset modes run the stages in the correct order:

```text
load training split
  -> fit preprocessor
  -> SSL pretraining
  -> save frozen encoder
  -> train task head
  -> load test split
  -> evaluate
  -> generate plots
```

## 8. Verification Performed

- Python syntax compilation passed for all modified modules.
- Dataset-loader unit tests passed:
  - KDD headerless parsing and exact file selection
  - CIC deterministic, stratified, disjoint split
  - standardized label creation
  - label metadata removal from feature tensors
- Real local KDD files loaded successfully:
  - training: 494,021 rows
  - test: 311,029 rows
- A real CICIDS2017 DDoS CSV loaded successfully:
  - training: 180,596 rows
  - test: 45,149 rows

## 9. Dataset References

- KDD Cup 99 archive:
  <https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>
- UCI KDD Cup 99 dataset record:
  <https://archive.ics.uci.edu/dataset/130/kdd+cup+1999+data>
- University of New Brunswick CICIDS2017 description:
  <https://www.unb.ca/cic/datasets/ids-2017.html>
