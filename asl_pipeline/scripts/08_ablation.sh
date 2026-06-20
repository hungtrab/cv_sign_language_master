#!/usr/bin/env bash
# Step 8: Component ablation — evaluate all swap variants
# Run after steps 01-07 complete.
set -e

DATASET="${DATASET:-data/asl_alphabet/test_split}"
OUTPUT="${OUTPUT:-outputs}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 8: Component Ablation ==="
echo "Dataset: ${DATASET}"

# Level 1: Baselines (already done in step 05, but re-run to be safe)
BASELINES="raw_siglip mediapipe_crop_resnet18 landmark_mlp enhancement_clahe_resnet18"

# Level 2: Component swaps
SWAPS="raw_resnet18 mediapipe_crop_vit mediapipe_landmarks_svm mediapipe_landmarks_rf no_enhance_resnet18 enhancement_gamma_resnet18 enhancement_sharpen_resnet18 enhancement_denoise_resnet18 enhancement_clahe_vit enhancement_gamma_vit"

ALL_PIPES="$BASELINES $SWAPS"

for PIPE in $ALL_PIPES; do
    # Skip if already evaluated
    if [ -f "${OUTPUT}/metrics/${PIPE}_metrics.json" ]; then
        echo "  [skip] ${PIPE} (already done)"
        continue
    fi

    echo ""
    echo "--- ${PIPE} ---"
    $PYTHON evaluate.py --dataset "${DATASET}" --pipeline "${PIPE}" --output "${OUTPUT}" 2>&1 | tail -3 || echo "  WARNING: ${PIPE} eval failed"

    $PYTHON benchmark_robustness.py --dataset "${DATASET}" --pipeline "${PIPE}" --output "${OUTPUT}" 2>&1 | tail -3 || echo "  WARNING: ${PIPE} robustness failed"
done

echo ""
echo "=== Regenerating comparison ==="
$PYTHON compare_pipelines.py --metrics-dir "${OUTPUT}/metrics" --output-dir "${OUTPUT}/figures" 2>&1 | tail -15

# Regenerate robustness summary from all JSONs
$PYTHON -c "
import json, csv
from pathlib import Path
with open('${OUTPUT}/metrics/robustness_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pipeline','corruption','severity','accuracy','macro_f1','accuracy_drop','relative_accuracy_drop','mean_confidence','unknown_rate','hand_detection_failure_rate','mean_latency_ms','fps','num_samples'])
    for rj in sorted(Path('${OUTPUT}/metrics').glob('*_robustness.json')):
        pipe = rj.stem.replace('_robustness','')
        data = json.load(open(rj))
        ca = data.get('clean',{}).get('accuracy',0)
        for n, m in data.items():
            d = round(ca - m.get('accuracy',0), 4) if n != 'clean' else 0
            rd = round(d / max(ca, 1e-6), 4) if n != 'clean' else 0
            w.writerow([pipe, n, m.get('severity',''), m.get('accuracy',0), m.get('macro_f1',0), d, rd, m.get('mean_confidence',0), m.get('unknown_rate',0), m.get('hand_detection_failure_rate',0), m.get('mean_latency_ms',0), m.get('fps',0), m.get('num_samples',0)])
print('robustness_summary.csv regenerated')
"

echo ""
echo "=== Ablation Done ==="
echo "Total pipelines evaluated: $(ls ${OUTPUT}/metrics/*_metrics.json 2>/dev/null | wc -l)"
