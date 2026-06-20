#!/usr/bin/env bash
# Step 4: Train landmark MLP classifier
set -e

PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 4: Train Landmark MLP ==="

$PYTHON train_landmark_mlp.py \
    --features data/landmarks.csv \
    --output weights/landmark_mlp.pkl

echo "Done. Model at weights/landmark_mlp.pkl"
