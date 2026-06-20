#!/usr/bin/env bash
# Step 6: Run robustness benchmarks
set -e

DATASET="${DATASET:-data/asl_alphabet}"
OUTPUT="${OUTPUT:-outputs}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 6: Robustness Benchmarks ==="

PIPELINES="mediapipe_resnet18 raw_hf landmark_mlp enhancement_clahe_resnet18"

for PIPE in $PIPELINES; do
    echo ""
    echo "--- Robustness: ${PIPE} ---"
    $PYTHON benchmark_robustness.py \
        --dataset "${DATASET}/test" \
        --pipeline "${PIPE}" \
        --output "${OUTPUT}" \
        2>&1 || echo "WARNING: ${PIPE} robustness failed, continuing..."
done

echo ""
echo "Done. Results at ${OUTPUT}/metrics/robustness_summary.csv"
