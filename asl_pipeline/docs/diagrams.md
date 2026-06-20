# Pipeline Diagrams

## Diagram 1 — Overall Modular Framework

```
┌──────────────────────────────────────────────────────────────┐
│  Input: RGB image / webcam frame                             │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Representation Module      │
        │  (interchangeable)          │
        │                             │
        │  • raw_image                │
        │  • mediapipe_crop           │
        │  • mediapipe_landmarks      │
        │  • enhancement (CLAHE/γ)    │
        │  • yolo_crop                │
        │  • segmentation_mask        │
        │  • fusion_rgb_landmarks     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Recognition Module         │
        │  (interchangeable)          │
        │                             │
        │  • resnet18_asl             │
        │  • hf_image_classifier      │
        │  • torchvision_classifier   │
        │  • landmark_mlp             │
        │  • fusion_classifier        │
        │  • yolo_direct_detector     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Post-processing            │
        │                             │
        │  • confidence threshold     │
        │  • Unknown rejection        │
        │  • majority vote smoothing  │
        │  • calibration              │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Output: A-Z prediction     │
        └─────────────────────────────┘
```

## Diagram 2 — Pipeline Families (P0–P7)

```
P1: Raw image ──────────→ CNN/ViT classifier ────→ A-Z

P2: RGB frame ──→ Hand detector ──→ Crop ──→ CNN classifier ──→ A-Z

P3: RGB frame ──→ Pose estimator ──→ 21 landmarks ──→ Normalize ──→ MLP ──→ A-Z

P4: RGB frame ──→ YOLO direct detector ──→ bbox + A-Z

P5: RGB frame ──→ Segmentation ──→ Mask ──→ Crop ──→ CNN classifier ──→ A-Z

P6: RGB frame ──→ Enhancement ──→ CNN/ViT classifier ──→ A-Z

P7: RGB frame ──→ ┌─ Crop ──→ CNN feature  ─┐
                   │                          ├──→ Fusion classifier ──→ A-Z
                   └─ Landmarks ──→ MLP feat ─┘
```

## Diagram 3 — Component Replacement Matrix

```
┌─────────────────────┬──────────────────┬──────────────────────┬────────────────────────┐
│ Component           │ Lightweight      │ Stronger             │ Research / Extension   │
├─────────────────────┼──────────────────┼──────────────────────┼────────────────────────┤
│ Raw classifier      │ ResNet18         │ EfficientNet/ConvNeXt│ ViT / Swin / SigLIP   │
│ Hand detection      │ MediaPipe        │ YOLOv8 / YOLO11     │ RT-DETR / custom       │
│ Hand pose           │ MediaPipe        │ MMPose / OpenPose    │ YOLO11 hand-pose       │
│ Segmentation        │ YOLO11-seg       │ U-Net / DeepLabV3+   │ SAM / SAM2             │
│ Enhancement         │ CLAHE / Gamma    │ Denoising / Deblur   │ Super-resolution       │
│ Landmark classifier │ MLP              │ 1D-CNN               │ Transformer encoder    │
│ Direct detection    │ YOLOv8           │ YOLO11               │ RT-DETR                │
│ Fusion              │ CNN + MLP        │ ViT + MLP            │ Multimodal transformer │
│ Post-processing     │ Threshold        │ Majority vote        │ Calibration            │
└─────────────────────┴──────────────────┴──────────────────────┴────────────────────────┘
```

## Diagram 4 — Dataset Roles

```
┌─────────────────────────────┬────────────────────────────────────────┐
│ Dataset                     │ Role                                   │
├─────────────────────────────┼────────────────────────────────────────┤
│ ASL Alphabet (Kaggle/HF)    │ Main A-Z classification (train/test)  │
│ Sign Language MNIST          │ Sanity check only (24 classes)        │
│ Roboflow ASL Letters         │ YOLO direct A-Z detection             │
│ Ultralytics Hand Keypoints   │ YOLO hand-pose training               │
│ HaGRID / EgoHands / etc.    │ Auxiliary hand detection/segmentation │
│ Real webcam validation set   │ Demo validation (5-10 per letter)     │
└─────────────────────────────┴────────────────────────────────────────┘
```

## Diagram 5 — Evaluation Protocol

```
Dataset
  │
  ├──→ Select pipeline (--pipeline mediapipe_resnet18)
  │
  ├──→ Clean evaluation
  │      ├── top-1 accuracy, top-3 accuracy
  │      ├── macro-F1, per-class F1
  │      ├── confusion matrix
  │      ├── confidence stats
  │      ├── latency / FPS
  │      └── hand detection failure rate
  │
  ├──→ Robustness evaluation
  │      ├── low_light, overexposure, motion_blur
  │      ├── gaussian_noise, rotation, scale_shift
  │      ├── partial_occlusion
  │      ├── accuracy drop per corruption
  │      └── relative accuracy drop
  │
  ├──→ Webcam validation
  │      ├── frame accuracy
  │      ├── majority vote accuracy
  │      ├── label switch rate
  │      ├── stable prediction delay
  │      └── unknown rate
  │
  └──→ Compare pipelines
         ├── accuracy comparison chart
         ├── latency comparison chart
         ├── robustness comparison chart
         ├── pipeline summary table
         └── failure case analysis
```

## Diagram 6 — Failure Case Categories

```
┌──────────────────────────────────────────────────┐
│  Failure Cases                                    │
├──────────────────────────────────────────────────┤
│                                                   │
│  hand_not_detected                                │
│    └── MediaPipe/YOLO fails to find a hand        │
│                                                   │
│  crop_failure                                     │
│    └── Crop misses fingers or is too tight        │
│                                                   │
│  low_light                                        │
│    └── Dark image degrades features               │
│                                                   │
│  background_clutter                               │
│    └── Cluttered background confuses classifier   │
│                                                   │
│  occlusion                                        │
│    └── Part of hand is hidden                     │
│                                                   │
│  similar_hand_shape                               │
│    └── A/S/E, M/N/T, U/V/R, G/H confusion        │
│                                                   │
│  motion_based_letter                              │
│    └── J and Z require motion (out of scope)      │
│                                                   │
│  low_confidence                                   │
│    └── Model uncertain, below threshold           │
│                                                   │
│  model_confident_but_wrong                        │
│    └── High confidence, wrong prediction          │
│                                                   │
└──────────────────────────────────────────────────┘
```
