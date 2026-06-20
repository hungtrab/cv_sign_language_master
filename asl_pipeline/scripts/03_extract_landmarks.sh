#!/usr/bin/env bash
# Step 3: Extract MediaPipe hand landmarks from training images
set -e

DATASET="${DATASET:-data/asl_alphabet}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 3: Extract Landmarks ==="
echo "Dataset: ${DATASET}/train"

$PYTHON extract_landmarks.py \
    --dataset "${DATASET}/train" \
    --output data/landmarks.csv

LINES=$(wc -l < data/landmarks.csv)
echo "Extracted landmarks: ${LINES} rows in data/landmarks.csv"
echo "Done."
