#!/usr/bin/env bash
# run_all.sh — Run every step in sequence.
# Each step is a standalone script in scripts/ that can be run independently.
#
# Usage:
#   ./run_all.sh                    # run everything
#   ./scripts/05_evaluate_all.sh    # run just evaluation
#
# Environment variables:
#   DATASET=data/asl_alphabet   (default)
#   OUTPUT=outputs              (default)
#   PYTHON=.venv/bin/python     (default)

set -e

export DATASET="${DATASET:-data/asl_alphabet}"
export OUTPUT="${OUTPUT:-outputs}"
export PYTHON="${PYTHON:-.venv/bin/python}"

echo "============================================"
echo "  ASL Pipeline — Full Evaluation Run"
echo "============================================"
echo "DATASET: ${DATASET}"
echo "OUTPUT:  ${OUTPUT}"
echo ""

bash scripts/01_setup.sh
bash scripts/02_download_data.sh
bash scripts/03_extract_landmarks.sh
bash scripts/04_train_mlp.sh
bash scripts/05_evaluate_all.sh
bash scripts/06_robustness.sh
bash scripts/07_compare.sh

echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"
echo ""
echo "Key outputs:"
echo "  ${OUTPUT}/tables/pipeline_summary.csv"
echo "  ${OUTPUT}/metrics/robustness_summary.csv"
echo "  ${OUTPUT}/figures/pipeline_accuracy_comparison.png"
echo "  ${OUTPUT}/figures/pipeline_robustness_comparison.png"
