#!/usr/bin/env bash
# Step 5: Evaluate all pipelines on clean test set
set -e

DATASET="${DATASET:-data/asl_alphabet}"
OUTPUT="${OUTPUT:-outputs}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 5: Evaluate All Pipelines ==="

PIPELINES="mediapipe_resnet18 raw_hf landmark_mlp enhancement_clahe_resnet18 enhancement_gamma_resnet18"

for PIPE in $PIPELINES; do
    echo ""
    echo "--- Evaluating: ${PIPE} ---"
    $PYTHON evaluate.py \
        --dataset "${DATASET}/test" \
        --pipeline "${PIPE}" \
        --output "${OUTPUT}" \
        2>&1 || echo "WARNING: ${PIPE} failed, continuing..."
done

echo ""
echo "Done. Metrics at ${OUTPUT}/metrics/"
