"""Download and prepare the ASL Alphabet dataset for evaluation.

Supports:
  1. HuggingFace Datasets (grassknoted/asl-alphabet) — auto-download
  2. Kaggle CLI — if kaggle is installed
  3. Manual — prints instructions

Usage:
    python download_data.py
    python download_data.py --method huggingface
    python download_data.py --method kaggle
    python download_data.py --output data/asl_alphabet
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from asl.utils.logging import get_logger

log = get_logger("asl.download_data")

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def download_huggingface(output_dir: Path):
    """Download ASL Alphabet from HuggingFace Datasets."""
    log.info("Downloading ASL Alphabet from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        log.info("Installing 'datasets' library...")
        os.system(f"{os.sys.executable} -m pip install datasets")
        from datasets import load_dataset

    ds = load_dataset("grassknoted/asl-alphabet", split="train")

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    for cls in AZ_CLASSES:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (test_dir / cls).mkdir(parents=True, exist_ok=True)

    log.info(f"Saving {len(ds)} images to {output_dir}...")
    counts = {c: 0 for c in AZ_CLASSES}
    for i, sample in enumerate(ds):
        label = sample["label"]
        image = sample["image"]

        label_name = ds.features["label"].int2str(label)
        if label_name.upper() not in AZ_CLASSES:
            continue

        cls = label_name.upper()
        counts[cls] += 1

        # 85% train, 15% test
        if counts[cls] % 7 == 0:
            dest = test_dir / cls / f"{cls}_{counts[cls]:05d}.jpg"
        else:
            dest = train_dir / cls / f"{cls}_{counts[cls]:05d}.jpg"

        image.save(str(dest))

        if (i + 1) % 5000 == 0:
            log.info(f"  {i + 1} images processed...")

    log.info(f"Done. Train: {train_dir}, Test: {test_dir}")
    for cls in AZ_CLASSES:
        n_train = len(list((train_dir / cls).glob("*")))
        n_test = len(list((test_dir / cls).glob("*")))
        log.info(f"  {cls}: {n_train} train, {n_test} test")


def download_kaggle(output_dir: Path):
    """Download ASL Alphabet from Kaggle."""
    log.info("Downloading ASL Alphabet from Kaggle...")
    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "grassknoted/asl-alphabet",
         "--unzip", "-p", str(output_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"Kaggle download failed: {result.stderr}")
        raise RuntimeError("Kaggle CLI failed. Is it installed and authenticated?")

    # Reorganize to expected structure
    kaggle_train = output_dir / "asl_alphabet_train" / "asl_alphabet_train"
    if kaggle_train.exists():
        for cls_dir in kaggle_train.iterdir():
            if cls_dir.is_dir() and cls_dir.name.upper() in AZ_CLASSES:
                dest = output_dir / "train" / cls_dir.name.upper()
                dest.mkdir(parents=True, exist_ok=True)
                for f in cls_dir.glob("*"):
                    shutil.move(str(f), str(dest / f.name))
        shutil.rmtree(str(output_dir / "asl_alphabet_train"), ignore_errors=True)

    kaggle_test = output_dir / "asl_alphabet_test" / "asl_alphabet_test"
    if kaggle_test.exists():
        for cls_dir in kaggle_test.iterdir():
            if cls_dir.is_dir() and cls_dir.name.upper() in AZ_CLASSES:
                dest = output_dir / "test" / cls_dir.name.upper()
                dest.mkdir(parents=True, exist_ok=True)
                for f in cls_dir.glob("*"):
                    shutil.move(str(f), str(dest / f.name))
        shutil.rmtree(str(output_dir / "asl_alphabet_test"), ignore_errors=True)

    log.info(f"Done. Dataset at {output_dir}")


def print_manual_instructions(output_dir: Path):
    print(f"""
Manual download instructions:

1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Download and unzip to: {output_dir}
3. Ensure the structure is:
   {output_dir}/
     train/
       A/  *.jpg
       B/  *.jpg
       ...
       Z/  *.jpg
     test/
       A/  *.jpg
       ...
       Z/  *.jpg

4. Or use HuggingFace:
   python download_data.py --method huggingface
""")


def main():
    parser = argparse.ArgumentParser(description="Download ASL Alphabet dataset")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "huggingface", "kaggle", "manual"],
                        help="Download method")
    parser.add_argument("--output", default="data/asl_alphabet",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Check if data already exists
    train_dir = output_dir / "train"
    if train_dir.exists() and any(train_dir.iterdir()):
        existing = len(list(train_dir.glob("*/*.jpg"))) + len(list(train_dir.glob("*/*.png")))
        log.info(f"Data already exists at {output_dir} ({existing} images). Skipping download.")
        return

    if args.method == "manual":
        print_manual_instructions(output_dir)
        return

    if args.method == "huggingface" or args.method == "auto":
        try:
            download_huggingface(output_dir)
            return
        except Exception as e:
            log.warning(f"HuggingFace download failed: {e}")
            if args.method == "huggingface":
                raise

    if args.method == "kaggle" or args.method == "auto":
        try:
            download_kaggle(output_dir)
            return
        except Exception as e:
            log.warning(f"Kaggle download failed: {e}")
            if args.method == "kaggle":
                raise

    print_manual_instructions(output_dir)


if __name__ == "__main__":
    main()
