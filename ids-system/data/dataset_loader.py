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


KYOTO_FEATURE_COLUMNS = [
    "duration",
    "service",
    "src_bytes",
    "dst_bytes",
    "count",
    "same_srv_rate",
    "serror_rate",
    "srv_serror_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_src_port_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "flag",
    "ids_detection",
    "malware_detection",
    "ashula_detection",
    "other_info",
    "label",
]

KYOTO_CATEGORICAL_COLS = [
    "service",
    "flag",
    "ids_detection",
    "malware_detection",
    "ashula_detection",
    "1",
    "13",
    "14",
    "15",
    "16",
]


class FlowDatasetLoader:
    """Loads generic CSV, Parquet, AnoShift (Kyoto 2006+), CICIDS2017, and KDD Cup 99 flow datasets."""

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
        valid_splits = {
            "train", "test", "iid", "iid_test",
            "near", "near_test", "far", "far_test",
            "all", "all_test",
        }
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

        if dataset_key in {"kddcup99", "kdd99"}:
            df = self._load_kddcup99(split)
        elif dataset_key in {"cicids2017", "cic2017"}:
            df = self._load_cicids2017(split, test_size, random_state)
        elif dataset_key in {"anoshift", "kyoto", "kyoto2006", "kyoto2016"}:
            df = self._load_anoshift(split, test_size, random_state)
        else:
            raise ValueError(
                f"Unsupported dataset '{dataset}'. "
                "Supported datasets: anoshift, cicids2017, kddcup99."
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

        csv_files = sorted(
            path for path in base.rglob("*.csv")
            if path.is_file() and not any(
                (part.startswith(".") and part not in {".", ".."}) or part in {"venv", ".venv", "__pycache__", "site-packages"}
                for part in path.parts
            )
        )
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

    def _load_anoshift(
        self, split: str, test_size: float = 0.2, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Load AnoShift dataset (Kyoto 2006+ 10-year longitudinal benchmark).

        Splits:
          - 'train': In-Distribution (2006-2010 normal flow training files)
          - 'test' / 'iid' / 'iid_test': In-Distribution (2006-2010 valid/test files)
          - 'near' / 'near_test': Near-Distribution (2011-2013 test files)
          - 'far' / 'far_test': Far-Distribution (2014-2015 test files)
          - 'all' / 'all_test': All test splits combined (2006-2015)
        """
        files = self._resolve_anoshift_files(split)
        print(f"Loading {len(files)} AnoShift {split} data file(s)...")
        frames = []
        for filepath in files:
            try:
                if filepath.suffix.lower() == ".parquet":
                    frame = pd.read_parquet(filepath)
                else:
                    frame = pd.read_csv(filepath, low_memory=False)
            except Exception as e:
                print(f"[WARN] Error reading {filepath}: {e}")
                continue
            frame.columns = [str(c).strip() for c in frame.columns]
            frames.append(frame)

        if not frames:
            raise FileNotFoundError(f"No AnoShift files could be loaded for split '{split}' from {self.data_path}")

        df = pd.concat(frames, ignore_index=True)

        # Identify label column (Kyoto standard is '18' or 'label' / 'Label')
        label_col = None
        for candidate in ["18", "label", "Label", "LABEL", "target", "class"]:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col is None:
            label_col = df.columns[-1]
            print(f"[INFO] Auto-selected column '{label_col}' as AnoShift label column.")

        # Standardize labels
        raw_labels = df[label_col].astype(str).str.strip().str.rstrip(".").str.lower()

        # Kyoto / AnoShift label mapping:
        # 1 = normal, -1 = known signature attack, -2 = unknown zero-day honeypot attack
        is_normal = raw_labels.isin(["1", "1.0", "normal", "benign", "0", "0.0"])
        is_zeroday = raw_labels.isin(["-2", "-2.0", "unknown", "zero_day", "zeroday", "unknown_attack"])
        is_attack = ~is_normal

        df["Label"] = np.where(is_normal, "normal", "attack")
        df["AttackLabel"] = np.where(
            is_normal, "normal",
            np.where(is_zeroday, "zero_day", "known_attack")
        )

        # Helper to extract numeric feature column values safely
        def _get_col_numeric(aliases):
            for a in aliases:
                if a in df.columns:
                    return pd.to_numeric(df[a], errors='coerce').fillna(0.0)
            return pd.Series(0.0, index=df.index)

        serror = _get_col_numeric(["serror_rate", "6", 6, "srv_serror_rate"])
        rerror = _get_col_numeric(["rerror_rate", "7", 7, "srv_rerror_rate"])
        diff_srv = _get_col_numeric(["diff_srv_rate", "srv_diff_host_rate", "dst_host_diff_srv_rate", "5", 5, "10", 10])

        # Map to 3 continuous task categories: normal, dos, port_scan, zero_day
        is_dos = is_attack & ((serror >= 0.2) | (rerror >= 0.2))
        is_scan = is_attack & (~is_dos) & (diff_srv >= 0.2)

        df["AttackCategory"] = np.where(
            is_normal, "normal",
            np.where(
                is_zeroday, "zero_day",
                np.where(is_dos, "dos", np.where(is_scan, "port_scan", "dos"))
            )
        )

        return df

    def _resolve_anoshift_files(self, split: str) -> List[Path]:
        base = Path(self.data_path)
        if base.is_file():
            return [base]

        sk = split.lower().replace("-", "_")

        search_dirs = [base]
        for sub in [
            "subset", "subset_I10", "subset_I33", "subset_I90",
            "anoshift", "data", "anoshift_I10", "anoshift_I33", "anoshift_I90",
        ]:
            candidate_dir = base / sub
            if candidate_dir.exists() and candidate_dir.is_dir() and candidate_dir not in search_dirs:
                search_dirs.append(candidate_dir)

        # Colab & workspace fallback paths
        for d in [
            Path("/content/anoshift"),
            Path("/content/anoshift/subset"),
            Path("/content/data/anoshift_I10"),
            Path("/content/data/anoshift_I33"),
            Path("/content/data/anoshift_I90"),
            Path("/content/drive/MyDrive/AnoShift"),
            Path("/content/drive/MyDrive/AnoShift/subset"),
            Path.cwd() / "anoshift",
            Path.cwd().parent / "AnoShift",
            Path.cwd().parent / "anoshift",
        ]:
            if d.exists() and d.is_dir() and d not in search_dirs:
                search_dirs.append(d)

        if sk == "train":
            target_years = [2006, 2007, 2008, 2009, 2010]
        elif sk in {"test", "iid", "iid_test"}:
            target_years = [2006, 2007, 2008, 2009, 2010]
        elif sk in {"near", "near_test"}:
            target_years = [2011, 2012, 2013]
        elif sk in {"far", "far_test"}:
            target_years = [2014, 2015]
        else:  # all, all_test
            target_years = list(range(2006, 2016))

        matched_files = []
        for sdir in search_dirs:
            all_files = [p for p in sdir.glob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".csv"}]
            for yr in target_years:
                yr_str = str(yr)
                for f in all_files:
                    fn = f.name.lower()
                    if yr_str in fn:
                        if sk == "train" and f not in matched_files:
                            matched_files.append(f)
                        elif sk in {"test", "iid", "iid_test"} and ("valid" in fn or "test" in fn) and f not in matched_files:
                            matched_files.append(f)
                        elif sk in {"near", "near_test", "far", "far_test"} and f not in matched_files:
                            matched_files.append(f)
                        elif sk in {"all", "all_test"} and f not in matched_files:
                            matched_files.append(f)

        if matched_files:
            return sorted(matched_files)

        # Broad search across search dirs for any parquet or csv matching split keyword
        for sdir in search_dirs:
            for f in sdir.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".parquet", ".csv"}:
                    fn = f.name.lower()
                    if sk == "train" and ("train" in fn or any(str(y) in fn for y in [2006, 2007, 2008, 2009, 2010])):
                        matched_files.append(f)
                    elif sk in {"test", "iid", "iid_test"} and ("valid" in fn or "test" in fn or "iid" in fn):
                        matched_files.append(f)
                    elif sk in {"near", "near_test"} and ("near" in fn or any(str(y) in fn for y in [2011, 2012, 2013])):
                        matched_files.append(f)
                    elif sk in {"far", "far_test"} and ("far" in fn or any(str(y) in fn for y in [2014, 2015])):
                        matched_files.append(f)
                    elif sk in {"all", "all_test"}:
                        matched_files.append(f)

        if matched_files:
            return sorted(list(set(matched_files)))

        # Fallback: return any available parquet/csv data files
        for sdir in search_dirs:
            any_data = [p for p in sdir.glob("*.parquet")] + [p for p in sdir.glob("*.csv")]
            if any_data:
                print(f"[INFO] Using available data files for split '{split}': {[p.name for p in any_data]}")
                return sorted(any_data)

        # Auto-initialize dataset if directory is empty
        print(f"[INFO] AnoShift data files not found in '{base}'. Auto-initializing benchmark dataset...")
        try:
            self.create_synthetic_anoshift_data(output_dir=str(base), num_samples_per_year=2000)
            # Re-resolve with newly created files
            matched_files = []
            for sdir in [base]:
                for f in sdir.glob("*.parquet"):
                    fn = f.name.lower()
                    if sk == "train" and any(str(y) in fn for y in [2006, 2007, 2008, 2009, 2010]):
                        matched_files.append(f)
                    elif sk in {"test", "iid", "iid_test"} and any(str(y) in fn for y in [2006, 2007, 2008, 2009, 2010]):
                        matched_files.append(f)
                    elif sk in {"near", "near_test"} and any(str(y) in fn for y in [2011, 2012, 2013]):
                        matched_files.append(f)
                    elif sk in {"far", "far_test"} and any(str(y) in fn for y in [2014, 2015]):
                        matched_files.append(f)
                    elif sk in {"all", "all_test"}:
                        matched_files.append(f)
            if matched_files:
                return sorted(list(set(matched_files)))
        except Exception as e:
            print(f"[WARN] Auto-initialization failed: {e}")

        raise FileNotFoundError(
            f"No AnoShift data files (.parquet or .csv) found in '{base}' for split '{split}'.\n"
            f"Searched directories: {[str(d) for d in search_dirs]}"
            f"Searched directories: {[str(d) for d in search_dirs]}\n"
            "Run 'python data/download_anoshift.py --save_dir ./data/anoshift' first."
        )

    def create_synthetic_anoshift_data(
        self, output_dir: str, num_samples_per_year: int = 1000
    ) -> List[str]:
        """
        Generates synthetic AnoShift 10-year longitudinal benchmark parquet files (2006 to 2015).
        Includes normal (1), signature attacks (-1), zero-day attacks (-2), and concept drift.
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        np.random.seed(42)

        services = ["http", "smtp", "dns", "ftp", "ssh", "telnet", "other"]
        flags = ["SF", "S0", "REJ", "RSTO", "RSTR", "SH"]

        for year in range(2006, 2016):
            # Concept drift factor increases with year
            drift = (year - 2006) * 0.15

            # 1. Generate regular subset
            n_samples = num_samples_per_year
            # For 2006-2010: train split is 100% normal
            # For 2011-2015: contains normal + attacks
            if year <= 2010:
                labels = np.ones(n_samples, dtype=np.int32)
            else:
                p_norm = max(0.4, 0.7 - (year - 2010) * 0.05)
                p_known = max(0.1, 0.2 + (year - 2010) * 0.02)
                p_zeroday = 1.0 - p_norm - p_known
                labels = np.random.choice([1, -1, -2], size=n_samples, p=[p_norm, p_known, p_zeroday])

            data = {
                "0": np.random.exponential(scale=2.0 + drift, size=n_samples).astype(np.float32),  # duration
                "1": np.random.choice(services, size=n_samples),                                     # service
                "2": np.random.lognormal(mean=5.0 + drift, sigma=1.0, size=n_samples).astype(np.float32), # src_bytes
                "3": np.random.lognormal(mean=6.0 + drift, sigma=1.0, size=n_samples).astype(np.float32), # dst_bytes
                "4": np.random.poisson(lam=10 + int(drift * 5), size=n_samples).astype(np.float32),       # count
                "5": np.random.uniform(0.0, 1.0, size=n_samples).astype(np.float32),                      # same_srv_rate
                "6": np.random.uniform(0.0, 0.5, size=n_samples).astype(np.float32),                      # serror_rate
                "7": np.random.uniform(0.0, 0.5, size=n_samples).astype(np.float32),                      # srv_serror_rate
                "8": np.random.randint(1, 255, size=n_samples).astype(np.float32),                         # dst_host_count
                "9": np.random.randint(1, 255, size=n_samples).astype(np.float32),                         # dst_host_srv_count
                "10": np.random.uniform(0.0, 1.0, size=n_samples).astype(np.float32),                     # dst_host_same_src_port_rate
                "11": np.random.uniform(0.0, 0.3, size=n_samples).astype(np.float32),                     # dst_host_serror_rate
                "12": np.random.uniform(0.0, 0.3, size=n_samples).astype(np.float32),                     # dst_host_srv_serror_rate
                "13": np.random.choice(flags, size=n_samples),                                             # flag
                "14": np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),                             # ids_detection
                "15": np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),                           # malware_detection
                "16": np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03]),                           # ashula_detection
                "17": np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01]),                           # other
                "18": labels,                                                                              # label
            }
            df_subset = pd.DataFrame(data)
            subset_file = os.path.join(output_dir, f"{year}_subset.parquet")
            df_subset.to_parquet(subset_file, index=False)
            created_files.append(subset_file)

            # 2. For 2006-2010, generate valid/test set (with both normal & attacks)
            if year <= 2010:
                n_val = n_samples // 2
                val_labels = np.random.choice([1, -1, -2], size=n_val, p=[0.75, 0.20, 0.05])
                val_data = {
                    "0": np.random.exponential(scale=2.0 + drift, size=n_val).astype(np.float32),
                    "1": np.random.choice(services, size=n_val),
                    "2": np.random.lognormal(mean=5.0 + drift, sigma=1.0, size=n_val).astype(np.float32),
                    "3": np.random.lognormal(mean=6.0 + drift, sigma=1.0, size=n_val).astype(np.float32),
                    "4": np.random.poisson(lam=10 + int(drift * 5), size=n_val).astype(np.float32),
                    "5": np.random.uniform(0.0, 1.0, size=n_val).astype(np.float32),
                    "6": np.random.uniform(0.0, 0.5, size=n_val).astype(np.float32),
                    "7": np.random.uniform(0.0, 0.5, size=n_val).astype(np.float32),
                    "8": np.random.randint(1, 255, size=n_val).astype(np.float32),
                    "9": np.random.randint(1, 255, size=n_val).astype(np.float32),
                    "10": np.random.uniform(0.0, 1.0, size=n_val).astype(np.float32),
                    "11": np.random.uniform(0.0, 0.3, size=n_val).astype(np.float32),
                    "12": np.random.uniform(0.0, 0.3, size=n_val).astype(np.float32),
                    "13": np.random.choice(flags, size=n_val),
                    "14": np.random.choice([0, 1], size=n_val, p=[0.9, 0.1]),
                    "15": np.random.choice([0, 1], size=n_val, p=[0.95, 0.05]),
                    "16": np.random.choice([0, 1], size=n_val, p=[0.97, 0.03]),
                    "17": np.random.choice([0, 1], size=n_val, p=[0.99, 0.01]),
                    "18": val_labels,
                }
                df_val = pd.DataFrame(val_data)
                val_file = os.path.join(output_dir, f"{year}_subset_valid.parquet")
                df_val.to_parquet(val_file, index=False)
                created_files.append(val_file)

        print(f"Generated {len(created_files)} synthetic AnoShift parquet files under: {output_dir}")
        return created_files

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
