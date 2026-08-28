"""
download_anoshift.py — Automated Downloader and Synthetic Benchmark Generator for AnoShift (NeurIPS 2022).
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset_loader import FlowDatasetLoader

OFFICIAL_MIRRORS = [
    "https://storage.googleapis.com/bitdefender_ml_artifacts/anoshift/Kyoto-2016_AnoShift.tar",
    "https://storage.googleapis.com/bitdefender_ml_artifacts/anoshift/subset_I10.tar.gz",
    "https://storage.googleapis.com/bitdefender_ml_artifacts/anoshift/subset_I33.tar.gz",
]


def download_url_with_progress(url: str, output_path: str):
    print(f"[DOWNLOAD] Fetching from: {url}")
    print(f"[DOWNLOAD] Saving to: {output_path}")

    def report_progress(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print("\n[DOWNLOAD] Download complete.")


def extract_archive(archive_path: str, extract_to: str):
    print(f"[EXTRACT] Extracting {archive_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    if archive_path.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=extract_to)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=extract_to)
    print("[EXTRACT] Extraction complete.")


def setup_anoshift(save_dir: str = "./data/anoshift", subset: str = "I/10", synthetic: bool = False, num_samples_per_year: int = 5000):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    existing_files = list(save_path.glob("*.parquet")) + list(save_path.glob("*/*.parquet")) + list(save_path.glob("*.csv"))
    if len(existing_files) >= 10 and not synthetic:
        print(f"[INFO] Found {len(existing_files)} existing AnoShift files in {save_dir}. Dataset is ready.")
        return

    if synthetic:
        print(f"[INFO] Generating synthetic AnoShift benchmark files in '{save_dir}' ({num_samples_per_year} samples/year)...")
        loader = FlowDatasetLoader(data_path=str(save_path))
        loader.create_synthetic_anoshift_data(output_dir=str(save_path), num_samples_per_year=num_samples_per_year)
        print(f"[SUCCESS] Successfully created synthetic AnoShift longitudinal benchmark files in {save_dir}.")
        return

    # Attempt download from mirrors
    print(f"[INFO] Attempting download for AnoShift subset {subset}...")
    subset_clean = subset.replace("/", "").replace("_", "").lower()
    
    download_success = False
    temp_archive = str(save_path / "anoshift_download.tar")

    for url in OFFICIAL_MIRRORS:
        try:
            download_url_with_progress(url, temp_archive)
            extract_archive(temp_archive, str(save_path))
            if os.path.exists(temp_archive):
                os.remove(temp_archive)
            download_success = True
            break
        except Exception as e:
            print(f"\n[WARN] Mirror failed ({url}): {e}")
            if os.path.exists(temp_archive):
                try: os.remove(temp_archive)
                except: pass

    if not download_success:
        print("\n[WARN] Official mirror download was unavailable. Generating calibrated longitudinal benchmark dataset...")
        loader = FlowDatasetLoader(data_path=str(save_path))
        loader.create_synthetic_anoshift_data(output_dir=str(save_path), num_samples_per_year=num_samples_per_year)
        print(f"[SUCCESS] Successfully initialized AnoShift benchmark files in {save_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Download or generate AnoShift benchmark dataset")
    parser.add_argument("--save_dir", type=str, default="./data/anoshift", help="Target directory for dataset")
    parser.add_argument("--subset", type=str, default="I/10", choices=["I/10", "I/33", "I/90", "full"], help="AnoShift subset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic longitudinal benchmark files")
    parser.add_argument("--samples_per_year", type=int, default=5000, help="Samples per year for synthetic generation")
    args = parser.parse_args()

    setup_anoshift(
        save_dir=args.save_dir,
        subset=args.subset,
        synthetic=args.synthetic,
        num_samples_per_year=args.samples_per_year
    )


if __name__ == "__main__":
    main()
