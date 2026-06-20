#!/usr/bin/env bash
# Step 7: Compare all pipelines and generate charts
set -e

OUTPUT="${OUTPUT:-outputs}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 7: Compare Pipelines ==="

$PYTHON compare_pipelines.py \
    --metrics-dir "${OUTPUT}/metrics" \
    --output-dir "${OUTPUT}/figures"

echo ""
echo "Done. Charts at ${OUTPUT}/figures/"
echo "Summary at ${OUTPUT}/tables/pipeline_summary.csv"
