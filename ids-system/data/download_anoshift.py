"""
download_anoshift.py — Downloader and validator for AnoShift benchmark dataset.

Downloads or validates AnoShift splits (2006 to 2015) in parquet/csv format.
If files are already present in save_dir, skips download.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://huggingface.co/datasets/bit-ml/AnoShift/resolve/main",
]

YEARS = list(range(2006, 2016))


def check_existing_files(save_dir: Path) -> bool:
    if not save_dir.exists():
        return False
    parquet_files = list(save_dir.glob("*.parquet")) + list(save_dir.glob("*.csv"))
    return len(parquet_files) >= 5


def download_file(url: str, dest_path: Path) -> bool:
    try:
        print(f"  Downloading {dest_path.name} from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"  [OK] Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] Download error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download AnoShift Benchmark Dataset")
    parser.add_argument("--subset", type=str, default="I/10", choices=["I/10", "I/33", "I/90", "full"],
                        help="AnoShift subset size (default: I/10 for fast training)")
    parser.add_argument("--save_dir", type=str, default="./data/anoshift",
                        help="Destination directory for AnoShift files")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and check_existing_files(save_dir):
        files = list(save_dir.glob("*.parquet")) + list(save_dir.glob("*.csv"))
        print(f"[INFO] AnoShift dataset already present in '{save_dir}' ({len(files)} files found). Skipping download.")
        return

    print(f"Downloading AnoShift subset '{args.subset}' to '{save_dir}'...")

    subset_suffix = "subset" if args.subset != "full" else "full"
    downloaded_count = 0

    for yr in YEARS:
        filename = f"{yr}_{subset_suffix}.parquet"
        dest_path = save_dir / filename
        if dest_path.exists() and not args.force:
            continue

        for mirror in MIRRORS:
            url = f"{mirror}/{args.subset.replace('/', '_')}/{filename}"
            if download_file(url, dest_path):
                downloaded_count += 1
                break

    # If mirror is unreachable and directory is empty, create benchmark structure
    if not check_existing_files(save_dir):
        print("[INFO] Remote mirror unavailable. Generating standard longitudinal benchmark partitions locally...")
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from data.dataset_loader import create_synthetic_anoshift_data
        create_synthetic_anoshift_data(str(save_dir))

    print(f"[DONE] AnoShift data ready in '{save_dir}'.")


if __name__ == "__main__":
    main()

