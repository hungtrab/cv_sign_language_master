# Sign Language Detection & Recognition

Computer Vision capstone — Semester 2025.2 — group of 3.

A real-time **two-stage** ASL alphabet recogniser:

1. **Stage 1 (Detection)** — YOLOv8 locates the hand in the camera frame.
2. **Stage 2 (Classification)** — a CNN (ResNet18 or MobileNetV2) classifies
   the cropped hand patch into one of 29 ASL classes (A–Z + space, delete,
   nothing).

The model runs at >30 FPS on a laptop CPU and supports an OpenCV webcam
demo, a Gradio web demo, and CLI evaluation.

```
            ┌──────────────────────────────────────────────────┐
            │  Webcam frame (OpenCV)                           │
            └──────────────────────┬───────────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  YOLOv8-nano                            │
              │  bbox(es) for hand(s)                   │
              └────────────────────┬────────────────────┘
                                   │ crop & resize 224×224
              ┌────────────────────▼────────────────────┐
              │  CNN classifier (ResNet18 / MobileNetV2)│
              │  29-class softmax                       │
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  Letter accumulation, save / clear      │
              │  Overlay on live frame                  │
              └─────────────────────────────────────────┘
```

## Quick start

```bash
make setup                         # venv + deps (PyTorch + ultralytics + opencv + gradio)
make data                          # creates data/ structure with download instructions
make train-detector                # trains YOLOv8 on prepared HaGRID hand data
make train-classifier              # trains the ASL classifier on Kaggle ASL Alphabet
make demo                          # live webcam demo (OpenCV window)
make demo-web                      # Gradio web UI on localhost:7860
make test                          # unit tests
```

Datasets must be placed under `data/`:

```
data/
├── asl_alphabet/                  # https://www.kaggle.com/datasets/grassknoted/asl-alphabet
│   ├── asl_alphabet_train/A/, B/, ..., Z/, space/, del/, nothing/
│   └── asl_alphabet_test/...
└── hagrid_hand/                   # https://github.com/hukenovs/hagrid (subset)
    ├── images/
    └── annotations/                # YOLO format
```

`make data` provides a script that prepares `data/yolo/` from HaGRID
annotations.

## What gets shipped

```
signlang/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/                       # YAML configs per training run
├── src/signlang/
│   ├── data/                      # ASL dataset, transforms, YOLO label prep
│   ├── detection/                 # YOLOv8 trainer + inference wrapper
│   ├── classification/            # CNN models, trainer, inference
│   ├── pipeline/                  # the two-stage pipeline
│   ├── demo/                      # OpenCV webcam + Gradio UI
│   ├── evaluation/                # metrics (accuracy, mAP, confusion matrix)
│   └── utils/                     # config, logging, viz helpers
├── scripts/                       # entry points (train / evaluate / demo)
├── tests/
└── docs/
    ├── ARCHITECTURE.md
    └── TASKS.md                   # 3-person split
```

## Team

| # | Member | ID | Owns |
|---|---|---|---|
| 1 | Trần Quang Hưng | 20235502 | YOLOv8 pipeline + dataset prep |
| 2 | Đỗ Đăng Vũ | 20235578 | Classifier (ResNet/MobileNet) + transforms |
| 3 | Lê Hoàng Tùng | 20235572 | Demo (OpenCV + Gradio) + evaluation + report |

See [`docs/TASKS.md`](docs/TASKS.md).

## Targets (from project proposal)

| Metric | Target |
|---|---|
| Hand detection mAP@0.5 | > 90% |
| Letter classification accuracy | > 95% |
| Inference speed (CPU) | ≤ 33 ms / frame |

VSL (Vietnamese Sign Language) is left as a stub — the model and pipeline
work for any 29-class CNN, so swapping in a VSL-trained checkpoint is a
config change. We don't ship a VSL model in v1.
