import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


KDD_FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

KDD_ATTACK_CATEGORIES = {
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "mailbomb": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "httptunnel": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "named": "r2l",
    "sendmail": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "worm": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
}


class FlowDatasetLoader:
    """Loads generic CSV, CICIDS2017, and KDD Cup 99 flow datasets."""

    def __init__(self, data_path: str, chunk_size: int = 100000):
        self.data_path = data_path
        self.chunk_size = chunk_size

    def load_csv(self, filepath: str, label_col: str = None) -> pd.DataFrame:
        """Load a standard CSV file that already contains a header row."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        self._validate_label_column(df, label_col)
        return df

    def load_dataset(
        self,
        dataset: str,
        split: str,
        label_col: str = "Label",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Load a supported raw dataset and expose standardized labels.

        Standard columns:
          - Label: binary "normal" or "attack"
          - AttackLabel: original normalized attack name
          - AttackCategory: normalized attack family
        """
        dataset_key = dataset.strip().lower().replace("-", "").replace("_", "")
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        if dataset_key in {"kddcup99", "kdd99"}:
            df = self._load_kddcup99(split)
        elif dataset_key in {"cicids2017", "cic2017"}:
            df = self._load_cicids2017(split, test_size, random_state)
        else:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. "
                "Supported datasets: cicids2017, kddcup99."
            )

        self._validate_label_column(df, label_col)
        print(
            f"Loaded {dataset} {split} split: {len(df):,} rows, "
            f"{len(df.columns):,} columns"
        )
        return df

    def _load_kddcup99(self, split: str) -> pd.DataFrame:
        filepath, is_partitioned = self._resolve_kdd_path(split)
        print(f"Loading KDD Cup 99 {split} data from {filepath}...")
        df = pd.read_csv(
            filepath,
            names=KDD_FEATURE_COLUMNS + ["AttackLabel"],
            header=None,
            compression="infer",
        )
        # Normalize attack labels first (needed for stratification)
        attack_labels = (
            df["AttackLabel"].astype(str).str.strip().str.rstrip(".").str.lower()
        )
        df["AttackLabel"] = attack_labels
        df["Label"] = np.where(attack_labels.eq("normal"), "normal", "attack")
        df["AttackCategory"] = attack_labels.map(self._kdd_attack_category)

        if is_partitioned:
            # Use stratified split to ensure all attack categories appear in both splits
            try:
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, stratify=df["AttackCategory"]
                )
            except ValueError:
                # Fallback: some categories may have only 1 sample, use AttackLabel
                print("[WARN] Stratified split on AttackCategory failed (rare classes). "
                      "Falling back to stratification on Label column.")
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, stratify=df["Label"]
                )
            if split == "train":
                df = train_df.reset_index(drop=True)
            else:
                df = test_df.reset_index(drop=True)
            print(f"[INFO] Stratified {split} subset: {len(df):,} samples")
            # Log class distribution for verification
            cat_counts = df["AttackCategory"].value_counts()
            print(f"[INFO] AttackCategory distribution in {split} split:\n{cat_counts.to_string()}")

        return df

    def _resolve_kdd_path(self, split: str) -> tuple[Path, bool]:
        base = Path(self.data_path)
        if base.is_file():
            return base, False

        candidates = {
            "train": [
                base / "kddcup.data_10_percent" / "kddcup.data_10_percent",
                base / "kddcup.data_10_percent_corrected",
                base / "kddcup.data_10_percent",
                base / "kddcup.data_10_percent.gz",
                base / "kddcup.data_10_percent.csv",
                base / "kddcup.data_10_percent_corrected.csv",
                base / "kddcup.data" / "kddcup.data",
                base / "kddcup.data",
                base / "kddcup.data.gz",
                base / "kddcup.data.csv",
                base / "kddcup.csv",
                base / "kddcup99.csv",
            ],
            "test": [
                base / "corrected" / "corrected",
                base / "corrected",
                base / "corrected.gz",
                base / "corrected.csv",
                base / "kddcup.data.corrected",
                base / "kddcup.data.corrected.csv",
                base / "kddcup.testdata.unlabeled.gz",
                base / "kddcup.testdata.unlabeled",
                base / "kddcup.testdata.unlabeled.csv",
            ],
        }[split]

        for candidate in candidates:
            if candidate.is_file():
                return candidate, False

        # Flexible fallback search if base directory exists
        if base.exists() and base.is_dir():
            all_files = [p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")]
            if split == "train":
                for f in all_files:
                    if "10_percent" in f.name.lower() or "train" in f.name.lower() or "kddcup.data" in f.name.lower():
                        print(f"[INFO] Auto-detected KDD train file: {f}")
                        return f, False
            else:
                for f in all_files:
                    if ("corrected" in f.name.lower() and "10_percent" not in f.name.lower()) or "test" in f.name.lower():
                        print(f"[INFO] Auto-detected KDD test file: {f}")
                        return f, False

        # Check parent directory, drive directories or current directory
        fallback_dirs = []
        for d in [
            base.parent if base.parent.exists() else None,
            Path("/content/drive/MyDrive/SSSL_UNFREEZED/KDDCUP99"),
            Path("/content/drive/MyDrive/SSSL_Based_IDS/KDDCUP99"),
            Path("/content/KDDCUP99"),
            Path("/content"),
            Path.cwd().parent / "KDDCUP99",
            Path.cwd(),
        ]:
            if d and d.exists() and d.is_dir() and d not in fallback_dirs:
                fallback_dirs.append(d)

        for search_dir in fallback_dirs:
            fallback_candidates = [
                search_dir / "kddcup.data_10_percent" / "kddcup.data_10_percent",
                search_dir / "kddcup.data_10_percent_corrected",
                search_dir / "kddcup.data_10_percent",
                search_dir / "kddcup.data_10_percent.gz",
                search_dir / "kddcup.data_10_percent.csv",
                search_dir / "kddcup.data_10_percent_corrected.csv",
                search_dir / "kddcup.data" / "kddcup.data",
                search_dir / "kddcup.data",
                search_dir / "kddcup.data.gz",
                search_dir / "kddcup.data.csv",
                search_dir / "kddcup.csv",
                search_dir / "kddcup99.csv",
            ] if split == "train" else [
                search_dir / "corrected" / "corrected",
                search_dir / "corrected",
                search_dir / "corrected.gz",
                search_dir / "corrected.csv",
                search_dir / "kddcup.data.corrected",
                search_dir / "kddcup.data.corrected.csv",
                search_dir / "kddcup.testdata.unlabeled.gz",
                search_dir / "kddcup.testdata.unlabeled",
                search_dir / "kddcup.testdata.unlabeled.csv",
            ]
            for cand in fallback_candidates:
                if cand.is_file():
                    print(f"[INFO] Auto-detected KDD {split} file: {cand}")
                    return cand, False

        # If dedicated test file not found, partition 20% test slice from available train file
        if split == "test":
            for search_dir in [base] + fallback_dirs:
                for train_cand in [
                    search_dir / "kddcup.data_10_percent_corrected",
                    search_dir / "kddcup.data_10_percent" / "kddcup.data_10_percent",
                    search_dir / "kddcup.data_10_percent",
                    search_dir / "kddcup.data_10_percent.gz",
                ]:
                    if train_cand.is_file():
                        print(f"[INFO] Dedicated test file not found. Auto-partitioning 20% test split from: {train_cand}")
                        return train_cand, True

        expected = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"The specified dataset directory '{base}' does not exist or does not contain KDD Cup 99 files.\n"
            f"Expected one of the following files:\n  - {expected}\n\n"
            f"Hint: If running in Google Colab with files uploaded directly to /content, pass '--data_path /content' or create the directory /content/KDDCUP99 and move the files there."
        )

    def _load_cicids2017(
        self, split: str, test_size: float, random_state: int
    ) -> pd.DataFrame:
        csv_files = self._find_cicids_csv_files()
        print(f"Loading {len(csv_files)} CICIDS2017 CSV file(s)...")
        selected_frames = []
        for filepath in csv_files:
            frame = pd.read_csv(filepath, low_memory=False)
            frame.columns = frame.columns.str.strip()
            if "Label" not in frame.columns:
                raise ValueError(f"CICIDS2017 label column not found in {filepath}")
            attack_labels = frame["Label"].astype(str).str.strip()
            frame["AttackLabel"] = attack_labels
            frame["AttackCategory"] = attack_labels.map(self._cic_attack_category)
            frame["Label"] = np.where(
                attack_labels.str.casefold().eq("benign"), "normal", "attack"
            )

            # Sequential (time-aware) split to prevent temporal session leakage across correlated flows
            split_idx = int(len(frame) * (1.0 - test_size))
            if split == "train":
                train_frame = frame.iloc[:split_idx]
                selected_frames.append(train_frame)
            else:
                test_frame = frame.iloc[split_idx:]
                selected_frames.append(test_frame)

        return pd.concat(selected_frames, ignore_index=True)

    def _find_cicids_csv_files(self) -> List[Path]:
        base = Path(self.data_path)
        if base.is_file() and base.suffix.lower() == ".csv":
            return [base]
        if not base.exists():
            raise FileNotFoundError(f"CICIDS2017 path not found: {base}")

        csv_files = sorted(path for path in base.rglob("*.csv") if path.is_file())
        if not csv_files:
            raise FileNotFoundError(f"No CICIDS2017 CSV files found under: {base}")
        return csv_files

    @staticmethod
    def _kdd_attack_category(attack_label: str) -> str:
        if attack_label == "normal":
            return "normal"
        return KDD_ATTACK_CATEGORIES.get(attack_label, "other")

    @staticmethod
    def _cic_attack_category(attack_label: str) -> str:
        label = str(attack_label).strip().casefold()
        if label == "benign":
            return "normal"
        if "portscan" in label:
            return "probe"
        if "ddos" in label or label.startswith("dos ") or "heartbleed" in label:
            return "dos"
        if "patator" in label:
            return "brute_force"
        if "web attack" in label:
            return "web_attack"
        if "infiltration" in label:
            return "infiltration"
        if "bot" in label:
            return "botnet"
        return "other"

    @staticmethod
    def _validate_label_column(df: pd.DataFrame, label_col: str):
        if label_col is not None and label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in dataset. "
                f"Columns available: {df.columns.tolist()}"
            )

    def create_synthetic_data(
        self, num_samples: int = 10000, num_features: int = 78
    ) -> pd.DataFrame:
        """Generate synthetic flow data for pipeline smoke tests."""
        print(
            f"Generating synthetic dataset with {num_samples} samples "
            f"and {num_features} features."
        )
        np.random.seed(42)

        features = np.random.randn(num_samples, num_features).astype(np.float32)
        labels = np.random.choice(
            ["Benign", "DoS", "PortScan", "Exfiltration"],
            size=num_samples,
            p=[0.7, 0.15, 0.1, 0.05],
        )

        df = pd.DataFrame(
            features, columns=[f"Feature_{i}" for i in range(num_features)]
        )
        df["Label"] = labels
        return df


if __name__ == "__main__":
    loader = FlowDatasetLoader(data_path=".")
    df = loader.create_synthetic_data(num_samples=100)
    print(f"Dataset Shape: {df.shape}")
    print(f"Label Distribution:\n{df['Label'].value_counts()}")
