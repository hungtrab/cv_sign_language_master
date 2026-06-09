# Architecture

```
              YAML (configs/*.yaml)            class names
                       │                            │
                       ▼                            ▼
  ┌────────────────────────────────────────────────────────┐
  │  scripts/run_demo.py                                  │
  └─┬──────────────────────────────────────────────────────┘
    │ build
    ▼
  ┌────────────────────┐    ┌────────────────────────────┐
  │ HandDetector       │    │ LetterClassifier           │
  │ (Ultralytics YOLO) │    │ (torchvision ResNet/MN-v2) │
  └─────────┬──────────┘    └──────────────┬─────────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
                ┌──────────────────────┐
                │ TwoStagePipeline     │
                │ - smoothing window   │
                │ - debounce cooldown  │
                │ - text buffer        │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       OpenCV demo               Gradio web demo
```

## Why two stages

Single-stage classification (whole frame -> letter) was tried in the v1
project (`src.pdf`) — performance fell apart with cluttered backgrounds.
By first detecting the hand and only feeding the cropped patch to the
CNN, the classifier sees a *clean* training distribution that matches
the Kaggle ASL Alphabet dataset (which is centred on the hand).

## Smoothing & debouncing

The classifier runs at every frame, so a single shaky prediction must
not corrupt the spelt buffer. Two filters:

* **Majority vote** over a sliding window of `smoothing_window` frames.
* **Cooldown** — after a letter is committed, suppress the same letter
  for `letter_repeat_cooldown_ms` so users don't accidentally double
  every key.

Special tokens:
* `space` → appends a space.
* `del` → drops the last character.
* `nothing` → no-op (the classifier saw no hand).

## Swapping the classifier

`models/multitask.py`-style — change `model.arch` in
`configs/classifier.yaml`:

```yaml
model:
  arch: mobilenet_v2     # or resnet18
```

Other backbones can be added with one helper in
`classification/models.py` (≈ 5 lines: load weights + replace head).

## VSL (Vietnamese Sign Language) — future work

The pipeline is class-agnostic: train another classifier on a VSL dataset
and point `paths.classifier_weights` at the new checkpoint. We include a
stub in `docs/VSL_TODO.md` with steps and dataset suggestions, but no
trained model is shipped in v1.
