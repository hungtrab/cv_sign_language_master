#!/usr/bin/env bash
# Step 2: Download ASL Alphabet dataset
set -e

DATASET="${DATASET:-data/asl_alphabet}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 2: Download Data ==="
echo "Output: ${DATASET}"

$PYTHON download_data.py --output "${DATASET}"

echo "Verifying..."
TRAIN_COUNT=$(find "${DATASET}/train" -type f 2>/dev/null | wc -l)
TEST_COUNT=$(find "${DATASET}/test" -type f 2>/dev/null | wc -l)
echo "  Train images: ${TRAIN_COUNT}"
echo "  Test images:  ${TEST_COUNT}"

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo "ERROR: No training images found. Check download."
    exit 1
fi

echo "Done."
