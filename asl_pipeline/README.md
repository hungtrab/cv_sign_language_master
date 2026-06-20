# Evaluating Modular Visual Representations for Static ASL Alphabet Recognition

Modular framework comparing multiple visual representations for static ASL A-Z recognition.

**Core question:** Do better hand representations actually improve downstream A-Z recognition, or do they only add complexity?

## Task Definition

```
Input:  RGB image / webcam frame containing one hand gesture
Output: one of 26 ASL alphabet classes: A-Z
```

This is static gesture recognition. J and Z are motion-based in real ASL but are treated as static 26-class targets for this demo.

## Pipeline Architecture

```
RGB image
→ Representation Module (interchangeable)
→ Recognition Module (interchangeable)
→ Post-processing (threshold, smoothing)
→ A-Z output
```

## Implemented Pipelines

| ID | Pipeline | Representation | Recognizer | Status |
|----|----------|---------------|------------|--------|
| P0 | `mediapipe_resnet18` | MediaPipe hand crop | ResNet18 (HuggingFace) | Ready |
| P1 | `raw_hf` | Raw image resize | HF image classifier (SigLIP) | Ready |
| P2 | `mediapipe_crop_vit` | MediaPipe hand crop | HF ViT/SigLIP | Ready |
| P3 | `landmark_mlp` | MediaPipe landmarks (42-dim) | MLP classifier | Ready |
| P6 | `enhancement_clahe_resnet18` | CLAHE enhancement | ResNet18 | Ready |
| P6 | `enhancement_gamma_resnet18` | Gamma correction | ResNet18 | Ready |
| P4 | `yolo11_direct_detection` | — | YOLO direct A-Z | Stub |
| P5 | `segmentation_resnet18` | Hand mask | ResNet18 | Stub |
| P7 | `fusion_rgb_landmarks` | RGB + landmarks | Fusion classifier | Stub |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start — Run Everything

```bash
./run_all.sh
```

This single command:
1. Downloads the ASL Alphabet dataset
2. Extracts landmarks from training images
3. Trains the landmark MLP classifier
4. Evaluates all pipelines on the clean test set
5. Runs robustness benchmarks (7 corruptions)
6. Generates comparison tables and charts

All results go to `outputs/`.

## Individual Commands

### Download Data

```bash
python download_data.py                        # auto-detect method
python download_data.py --method huggingface   # from HuggingFace
python download_data.py --method kaggle        # from Kaggle CLI
```

### Webcam Demo

```bash
python webcam_demo.py --pipeline mediapipe_resnet18
python webcam_demo.py --pipeline landmark_mlp
python webcam_demo.py --pipeline enhancement_clahe_resnet18
python webcam_demo.py --list-pipelines
```

### Static Image Prediction

```bash
python predict_image.py --image path/to/hand.jpg --pipeline mediapipe_resnet18
```

### Evaluate on Test Set

```bash
python evaluate.py --dataset data/asl_alphabet/test --pipeline mediapipe_resnet18
```

Outputs: `metrics.json`, `predictions.csv`, `confusion_matrix.png`, `per_class_f1.png`, `failure_summary.csv`

### Robustness Benchmark

```bash
python benchmark_robustness.py --dataset data/asl_alphabet/test --pipeline mediapipe_resnet18
```

Corruptions: low_light, overexposure, motion_blur, gaussian_noise, rotation, scale_shift, partial_occlusion

### Compare All Pipelines

```bash
python compare_pipelines.py --metrics-dir outputs/metrics
```

### Evaluate Webcam Recording

```bash
python evaluate_webcam_recording.py --video recording.mp4 --label A --pipeline mediapipe_resnet18
```

### Train

```bash
# Landmark MLP
python extract_landmarks.py --dataset data/asl_alphabet/train --output data/landmarks.csv
python train_landmark_mlp.py --features data/landmarks.csv

# Image classifier
python train_image_classifier.py --dataset data/asl_alphabet/train --model resnet18 --loss ce
```

## Component Replacement Matrix

| Component | Lightweight | Stronger | Research |
|-----------|-------------|----------|----------|
| Raw classifier | ResNet18 | EfficientNet / ConvNeXt | ViT / SigLIP |
| Hand detection | MediaPipe | YOLOv8 / YOLO11 | RT-DETR |
| Hand pose | MediaPipe Hand Landmarker | MMPose / OpenPose | YOLO11 hand-pose |
| Segmentation | YOLO11-seg | U-Net / DeepLabV3+ | SAM / SAM2 |
| Enhancement | CLAHE / Gamma | Denoising / Deblurring | Super-resolution |
| Landmark classifier | MLP | 1D-CNN | Transformer encoder |
| Direct detection | YOLOv8 | YOLO11 | RT-DETR |
| Fusion | CNN + MLP | ViT + MLP | Multimodal transformer |
| Post-processing | Threshold | Majority vote | Calibration |

## Evaluation Metrics

Classification: top-1 accuracy, top-3 accuracy, macro-F1, per-class F1, confusion matrix

Runtime: mean/median/p95 latency, FPS, model size

Robustness: accuracy under 7 corruptions, accuracy drop, relative drop

Webcam: label switch rate, unknown rate, stable prediction delay

## Output Structure

```
outputs/
├── metrics/          — JSON metrics per pipeline + robustness CSVs
├── predictions/      — CSV per-sample predictions
├── figures/          — Confusion matrices, bar charts, comparisons
├── tables/           — Pipeline summary, failure reports
├── failure_cases/    — Misclassified images
├── runs/             — Per-run config + environment
├── checkpoints/      — Trained model weights
└── logs/             — Training logs
```

## Limitations

This project formulates ASL alphabet recognition as a 26-class static image classification problem for demo purposes. In real ASL fingerspelling, J and Z are motion-based signs. Therefore, static recognition of J and Z should be interpreted as a simplified approximation rather than a complete linguistic model of ASL fingerspelling.
