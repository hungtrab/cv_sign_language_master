# CV Project Notes — Sign Language Detection & Recognition

Last reviewed: 2026-06-09

## Repository Status For Teammates

GitHub repository:

```text
https://github.com/hungtrab/cv_sign_language_master.git
```

This repo now includes source code, configs, notebooks, docs, tests, notes, and the two local demo checkpoints:

```text
runs/detector/weights/best.pt
runs/classifier_resnet18/best.pt
```

The checkpoint files are tracked with Git LFS because they are binary model weights. Anyone cloning the project should install Git LFS first:

```bash
git lfs install
git clone https://github.com/hungtrab/cv_sign_language_master.git
cd cv_sign_language_master
git lfs pull
```

If `git lfs pull` is skipped, the `.pt` files may be downloaded as tiny pointer files instead of real model weights, and the demo will fail when loading YOLO/classifier checkpoints.

## What Has Already Been Done

- Built a two-stage ASL recognition pipeline: webcam frame -> YOLO hand detector -> crop -> CNN classifier -> smoothing/debounce -> text buffer.
- Added YOLO detector wrapper with single-class `hand` checkpoint validation.
- Added CNN classifier inference wrapper.
- Added OpenCV webcam demo and Gradio demo entry points.
- Added smoothing, confidence threshold, margin threshold, cooldown, and neutral-frame re-arm logic to reduce repeated/spam letters.
- Added debug crop dumping with key `d` in the OpenCV demo.
- Added configs under `configs/`.
- Added training/evaluation scripts under `scripts/`.
- Added Colab notebooks for YOLO and classifier training under `notebooks/`.
- Added unit tests for config, metrics, and pipeline behavior under `tests/`.
- Added local demo weights for detector and classifier.

## How To Run From A Fresh Clone

Recommended environment:

```text
Python 3.10+ or 3.11 is safest.
Webcam access is needed for the OpenCV demo.
```

Setup:

```bash
git lfs install
git clone https://github.com/hungtrab/cv_sign_language_master.git
cd cv_sign_language_master
git lfs pull

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run OpenCV webcam demo:

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

or:

```bash
make demo
```

Run Gradio web demo:

```bash
python -m signlang.demo.gradio_app
```

or:

```bash
make demo-web
```

OpenCV demo keys:

```text
q -> quit
c -> clear current text buffer
s -> save current text to output.txt
d -> dump raw frame, crop, and prediction metadata to debug_crops/
```

If the webcam does not open, edit `camera_index` in `configs/demo.yaml`. Common values are `0`, `1`, or `2`.

## Current Project Shape

This project is a two-stage real-time ASL alphabet recognizer:

```text
webcam frame -> YOLOv8 hand detector -> crop hand -> CNN classifier -> smoothing/debounce -> text buffer
```

The architecture is sensible for a course demo. The detector and classifier are separated cleanly, and the runtime pipeline already has important demo safeguards:

- YOLO single-class `hand` detector.
- Classifier checkpoint loaded from `runs/classifier_resnet18/best.pt`.
- Confidence threshold and top-1/top-2 margin threshold.
- Majority-vote smoothing window.
- Neutral-frame re-arm logic to stop repeated text spam.
- Debug capture with key `d` to save raw frame, crop, prediction, top-5, confidence, and metadata.

Important files:

```text
README.md
configs/demo.yaml
configs/classifier.yaml
configs/detector.yaml
src/signlang/pipeline/two_stage.py
src/signlang/detection/detector.py
src/signlang/classification/inference.py
src/signlang/demo/webcam.py
notebooks/colab_retrain_yolov8s_singleclass.ipynb
notebooks/colab_train_hagrid_yolov8s.ipynb
notebooks/colab_train_classifier.ipynb
runs/detector/weights/best.pt
runs/classifier_resnet18/best.pt
```

## Current Checkpoints

Existing local weights:

```text
runs/detector/weights/best.pt        ~22 MB
runs/classifier_resnet18/best.pt     ~9.3 MB
```

The detector wrapper explicitly checks that the YOLO checkpoint is single-class:

```python
if self.class_names != {0: "hand"}:
    raise ValueError(...)
```

So if a YOLO file contains COCO classes or multiple unexpected classes, the demo should fail early instead of silently detecting wrong objects.

## Main Technical Risk

The biggest risk is still not code architecture. The biggest risk is **data distribution mismatch**.

The classifier was trained mostly on Kaggle ASL Alphabet images, which are usually:

- centered hand crops,
- controlled background,
- relatively clean lighting,
- fixed image style.

At runtime, the classifier receives YOLO crops from a real webcam, which can contain:

- partial hands,
- bad crops,
- background clutter,
- face/body regions if detector is wrong,
- different lighting and camera quality,
- different hand scale and orientation.

This explains why Kaggle validation can look perfect while the real demo still classifies poorly.

## Detector Notes

The detector is acceptable for a prepared demo if it produces stable hand boxes.

Things to watch:

- If the box is too large, the crop includes body/background and hurts classification.
- If the box is too tight, fingers are cut off and classification becomes unstable.
- If detection confidence is too low, increase `detection_conf_threshold` in `configs/demo.yaml`.
- If it detects non-hand regions, the classifier will still try to classify unless rejection logic catches it.

Current config:

```yaml
detection_conf_threshold: 0.5
```

For demo, test values around:

```text
0.45, 0.50, 0.60
```

Choose the one with the least false boxes while still detecting hands reliably.

## Classifier Notes

The classifier is the most fragile part.

Current runtime thresholds in `configs/demo.yaml`:

```yaml
classification_conf_threshold: 0.85
classification_margin_threshold: 0.20
smoothing_window: 5
letter_repeat_cooldown_ms: 600
require_neutral_between_letters: true
neutral_frames_to_reset: 3
```

This is a good starting point. The threshold should not be treated as universal; it must be calibrated on real webcam crops.

Recommended quick tuning:

- If it outputs too many wrong letters: raise confidence threshold, e.g. `0.90`.
- If it refuses too often: lower threshold slightly, e.g. `0.80`.
- If top-1 changes between similar classes: increase margin threshold, e.g. `0.25`.
- If it feels too slow to accept letters: reduce smoothing window to `3`.
- If letters repeat too easily: keep `require_neutral_between_letters: true`.

## Text Spam Prevention

The current pipeline already includes the correct idea:

```text
show sign -> stable prediction -> emit once -> hold sign -> no spam -> release/no-hand -> re-arm -> next sign
```

This is handled by:

- majority vote over `smoothing_window`,
- cooldown,
- neutral-frame re-arm,
- uncertain predictions do not re-arm.

The tests in `tests/test_pipeline.py` cover this logic. This is a strong point to mention in the report/demo because it shows the project handles real-time instability, not just per-frame classification.

## Debug Capture Is Important

The OpenCV demo supports key `d`:

```text
d -> save raw frame + crop + JSON metadata under debug_crops/
```

Use this whenever classification looks wrong. Do not guess from the final label only. Inspect:

- original frame,
- YOLO crop,
- predicted label,
- confidence,
- top-5 classes,
- bbox size.

Most errors will be obvious after looking at the crop.

## Demo Advice

For a reliable course demo, do not try to demonstrate all 29 classes live unless they are actually stable.

Recommended demo subset:

```text
A, B, C, L, O, V, Y, space, del, nothing
```

Pick signs that are visually distinct and work under the room lighting.

Before demo:

1. Check camera index in `configs/demo.yaml`.
2. Check lighting and background.
3. Run 5-10 target signs and note which ones are stable.
4. Avoid signs that classifier confuses.
5. Keep `mirror: false` unless the training/evaluation pipeline was designed for mirrored webcam input.
6. Use a simple target word, not a long sentence.

Possible demo phrase:

```text
A B C
```

or, if stable:

```text
LOVE
```

## What To Say In Presentation

Good technical framing:

- This is a two-stage detector-classifier pipeline.
- YOLO localizes the hand; CNN only sees a hand crop.
- This reduces background noise compared to whole-frame classification.
- Runtime predictions are smoothed and debounced to handle frame-level noise.
- The main limitation is dataset shift between Kaggle crops and real webcam crops.
- Future improvement is collecting real webcam-crop data and fine-tuning.

Avoid overclaiming:

- Do not claim it is deployable for general users.
- Do not rely only on Kaggle accuracy.
- Do not claim VSL support unless a VSL model is actually trained.

Safe claim:

```text
The system is suitable for a controlled course demo and provides a reusable architecture for sign recognition. Real deployment requires a webcam-crop dataset, stronger rejection logic, and end-to-end validation.
```

## Priority Improvements

### 1. Build a Webcam-Crop Dataset

Use the current YOLO detector to collect real crops from the actual demo pipeline.

Suggested layout:

```text
data/webcam_asl/
  A/
  B/
  C/
  ...
  space/
  del/
  nothing/
  unknown/
```

Minimum for quick improvement:

```text
50-100 images/class for demo subset
```

Better:

```text
200-300 images/class, multiple people/backgrounds/lighting
```

### 2. Fine-Tune, Do Not Retrain From Scratch

Start from current classifier checkpoint.

Recommended fine-tuning:

```text
lr = 1e-4 or 3e-5
epochs = 5-10
```

Data mix:

```text
70% Kaggle + 30% webcam crops
```

If enough webcam crops exist, increase webcam proportion.

### 3. Calibrate Thresholds

Use a real webcam validation set to tune:

```yaml
classification_conf_threshold
classification_margin_threshold
smoothing_window
neutral_frames_to_reset
```

Current guesses are reasonable but should be justified with validation data.

### 4. Add Unknown / Nothing Robustness

Include negative examples:

- no hand,
- invalid gesture,
- bad YOLO crop,
- face/body/background,
- partial hand.

These should map to `nothing` or `unknown`. Without this, the classifier is forced to output a letter too often.

### 5. End-to-End Evaluation

Report more than Kaggle accuracy:

- per-class accuracy on webcam crops,
- macro-F1,
- confusion matrix,
- false accept rate for `nothing` / `unknown`,
- detector failure rate,
- FPS / latency.

## Things Not Worth Doing First

Do not spend first effort on:

- bigger backbone,
- dynamic signs like J/Z,
- text-to-speech,
- mobile deployment,
- training YOLO again if boxes are already acceptable.

Data quality and classifier calibration matter more right now.

## Known Documentation Mismatches / Cleanup

Some docs still sound more optimistic than the current practical state.

Suggested wording:

```text
The model can run in real time in a controlled demo environment. Robust deployment requires additional real webcam-crop training and evaluation.
```

If report mentions `>95% classifier accuracy`, clarify whether it is:

- Kaggle validation accuracy, or
- real webcam-crop end-to-end accuracy.

Those are not the same metric.

## Commands

Common commands:

```bash
make test
make demo
make demo-web
python -m scripts.run_demo --config configs/demo.yaml
```

If using installed entry point:

```bash
signlang-demo --config configs/demo.yaml
```

## Final Recommendation

For the course deadline, focus on a controlled, honest demo:

1. Use the best single-class hand detector checkpoint.
2. Use a small stable sign subset.
3. Tune classification thresholds on the demo camera.
4. Use debug capture to understand failures.
5. Explain dataset shift clearly in the report.
6. Present webcam-crop fine-tuning as the next concrete improvement.

The project is architecturally fine. The next leap in quality comes from data and evaluation, not from rewriting the pipeline.
