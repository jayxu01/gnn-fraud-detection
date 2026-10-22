"""Download the Elliptic dataset (three CSVs) into data/raw/.

Uses the Kaggle CLI when available (requires kaggle.json credentials), otherwise
prints manual download steps.

Usage: python scripts/download_data.py
"""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

KAGGLE_DATASET = "ellipticco/elliptic-data-set"
FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def download_via_kaggle():
    """Download and unzip via the kaggle CLI."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
           "-p", str(RAW_DIR), "--unzip"]
    subprocess.run(cmd, check=True)


def main():
    if all((RAW_DIR / f).exists() for f in FILES):
        print("The three CSVs already exist, nothing to do.")
        return

    if shutil.which("kaggle"):
        print("kaggle CLI detected, downloading...")
        try:
            download_via_kaggle()
            print("Download complete.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"kaggle CLI download failed: {exc}")
    else:
        print("kaggle CLI not detected.")

    print("\nDownload manually instead:")
    print(f"  1. Open https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print("  2. Sign in to Kaggle and click Download")
    print("  3. Unzip and place these three files into data/raw/:")
    for f in FILES:
        print(f"     - {f}")
    print(f"\nTarget directory: {RAW_DIR}")


if __name__ == "__main__":
    main()
