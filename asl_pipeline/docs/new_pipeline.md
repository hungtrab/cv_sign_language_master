# New Pipeline Design: Enhancement × Representation × Classifier

## Research Question

> Does image enhancement improve downstream static ASL A-Z recognition, or does it only add complexity and potentially hurt performance?

## Pipeline Architecture

```
RGB Image
  │
  ▼
┌─────────────────────┐
│  Enhancement Module  │  ← 5 options (including "raw" = no enhancement)
│                     │
│  Raw (pass-through) │
│  CLAHE              │
│  Gamma Correction   │
│  Sharpening         │
│  Zero-DCE++         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────┐
│  Representation Module   │  ← 3 options
│                         │
│  MediaPipe Landmarks    │  → 21 keypoints → normalize → 42-dim vector
│  MMPose Landmarks       │  → 21 keypoints → normalize → 42-dim vector
│  YOLOv11 Hand Crop      │  → detect hand → crop → resize 224×224
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Classifier Module       │  ← depends on representation output
│                         │
│  For landmarks (42-dim):│
│    MLP                  │
│    XGBoost              │
│                         │
│  For image (224×224):   │
│    ResNet18             │
└─────────┬───────────────┘
          │
          ▼
    A-Z Prediction
```

## Why This Design

### Previous pipeline was flat

The old design treated each pipeline as a fixed combination:

```
mediapipe_crop_resnet18    = MediaPipe crop → ResNet18
raw_siglip                 = Raw image → SigLIP
landmark_mlp               = MediaPipe landmarks → MLP
enhancement_clahe_resnet18 = CLAHE → resize → ResNet18
```

This made it impossible to answer: **"Does CLAHE help MediaPipe detect hands better?"** because enhancement was only applied to the raw-image-to-classifier path, never before the hand detector.

### New pipeline chains enhancement BEFORE detection

```
Old:  RGB → CLAHE → resize → ResNet18              (enhancement replaces detection)
New:  RGB → CLAHE → MediaPipe detect → crop → MLP   (enhancement helps detection)
```

This is the correct experiment because in a real webcam demo, enhancement would be applied to the raw frame before any downstream processing.

## The 25-Run Experiment Grid

### Dimensions

| Dimension | Options | Count |
|-----------|---------|-------|
| Enhancement | Raw, CLAHE, Gamma, Sharpening, Zero-DCE++ | 5 |
| Representation + Classifier | MP→MLP, MP→XGBoost, MMPose→MLP, MMPose→XGBoost, YOLO→ResNet18 | 5 |
| **Total** | | **25** |

### Why these specific choices

#### Enhancement methods

| Method | Type | Rationale |
|--------|------|-----------|
| **Raw** | Baseline | Control group — no enhancement applied |
| **CLAHE** | Rule-based, contrast | Widely used in medical imaging and hand gesture papers. Enhances local contrast but can create artifacts in uniform regions |
| **Gamma Correction** | Rule-based, brightness | Simple global brightness adjustment. Less aggressive than CLAHE, fewer artifacts |
| **Sharpening** | Rule-based, edges | Enhances edges which could help finger-level keypoint detection. But amplifies noise |
| **Zero-DCE++** | Learned, low-light | Only 10K parameters. Trained with unsupervised loss (no paired data needed). Designed specifically for low-light enhancement without creating artifacts. The only method in this grid that learns image-adaptive curves rather than applying fixed transforms |

#### Representation methods

| Method | Output | Hand Detection | Rationale |
|--------|--------|---------------|-----------|
| **MediaPipe Hand Landmarker** | 42-dim vector (21 × x,y) | Built-in palm detector + landmark model | Industry standard. Fast on CPU. But sensitive to lighting and background |
| **MMPose RTMPose-Hand** | 42-dim vector (21 × x,y) | Full-image inference, no separate detector | Academic SOTA. More robust than MediaPipe because it does not require a separate palm detection step — it estimates keypoints on the entire image. But may produce noisier landmarks on cluttered backgrounds |
| **YOLOv11 Hand Detector** | 224×224 cropped image | YOLO object detection | Different paradigm — outputs an image crop, not landmarks. Allows using an image classifier (ResNet18) instead of a landmark classifier. Tests whether pixel-level information is more robust than geometric keypoints |

#### Classifiers

| Classifier | Input Type | Rationale |
|------------|-----------|-----------|
| **MLP** (256→128→64) | 42-dim features | Standard neural baseline for landmark classification. Fast training, well-understood |
| **XGBoost** | 42-dim features | Gradient-boosted trees. Often outperforms neural networks on structured/tabular data. Provides feature importance for interpretability |
| **ResNet18** | 224×224 image | CNN baseline for image classification. Used with YOLO crop because the crop is an image, not a feature vector |

### Why not all classifiers for all representations

Landmark representations output a 42-dimensional feature vector. Image classifiers (ResNet18) expect a 224×224 image. They are incompatible:

```
MediaPipe Landmarks → 42-dim vector → MLP ✓      ResNet18 ✗
MMPose Landmarks    → 42-dim vector → XGBoost ✓   ResNet18 ✗
YOLO Crop           → 224×224 image → ResNet18 ✓  MLP ✗
```

This is not a limitation — it reflects the fundamental difference between geometric (landmark-based) and appearance (image-based) recognition.

## Component Analysis

### Enhancement effects on hand detection

The key insight from our experiments: **enhancement applied before the hand detector changes the detection failure rate**, not just the classification accuracy.

```
Enhancement → Hand Detector → ...
     ↑                ↑
  Changes the       May detect
  input image       MORE or FEWER
                    hands depending
                    on enhancement
```

Traditional enhancements (CLAHE, Sharpening) can **increase** detection failure because they create artifacts that confuse the palm detector. Zero-DCE++ is designed to avoid this — it learns pixel-wise curves that brighten dark regions without creating false edges or color artifacts.

### Pose estimator comparison

MediaPipe and MMPose both output 21 hand keypoints, but they differ in detection strategy:

```
MediaPipe:
  Frame → Palm Detector → crop → Landmark Model → 21 keypoints
  (Two-stage: first find palm, then estimate landmarks on crop)

MMPose RTMPose-Hand:
  Frame → Single-stage → 21 keypoints
  (Direct regression on the full image or a provided bounding box)
```

This means:
- MediaPipe **fails completely** when the palm detector misses the hand
- MMPose **always produces keypoints** but they may be less precise on non-hand regions

The fail rate difference (MediaPipe ~17% vs MMPose ~6%) directly impacts real-world accuracy.

### Landmark classifier comparison

MLP and XGBoost receive the same 42-dim normalized input:

```
[x0, y0, x1, y1, ..., x20, y20]
   ↑
   Wrist-relative, scale-normalized to [-1, 1]
```

The normalization is critical — it removes position and scale variation, so the classifier only sees hand shape geometry.

XGBoost's advantage: it handles feature interactions naturally through tree splits, which maps well to "if finger 4 tip is above finger 4 base AND finger 3 is curled" type patterns in ASL gestures.

## Evaluation Protocol

### Metrics per pipeline

| Metric | Definition |
|--------|-----------|
| **Clean Accuracy** | Accuracy on detected images from the clean test set |
| **Real Accuracy** | Accuracy over ALL test images (undetected = wrong) |
| **Hand Detection Failure Rate** | % of images where the detector found no hand |
| **Low-Light Robustness** | Accuracy on the low-light corrupted test set |

### Why Real Accuracy ≠ Clean Accuracy

```
Clean Accuracy = correct predictions / detected images
Real Accuracy  = correct predictions / ALL images
```

A pipeline with 99% Clean Accuracy but 50% detection failure has only ~50% Real Accuracy. This is why detection failure rate is the most important metric for real-world deployment.

### Controlled comparisons

The grid enables three types of controlled ablation:

**1. Enhancement ablation** (fix representation+classifier, vary enhancement):
```
Raw    → MP → MLP  vs  CLAHE → MP → MLP  vs  Gamma → MP → MLP  vs  ...
```
Answers: Does enhancement help or hurt for this specific representation?

**2. Pose estimator ablation** (fix enhancement+classifier, vary representation):
```
Raw → MediaPipe → MLP  vs  Raw → MMPose → MLP
```
Answers: Which pose estimator is more robust?

**3. Classifier ablation** (fix enhancement+representation, vary classifier):
```
Raw → MP → MLP  vs  Raw → MP → XGBoost
```
Answers: Does the classifier choice matter on landmark features?

## Expected Findings

Based on reference data from initial experiments:

1. **Enhancement hurts detection more than it helps classification** — CLAHE and Sharpening roughly double the MediaPipe failure rate

2. **Zero-DCE++ is the exception** — as a learned enhancement, it reduces failure rate below the raw baseline while also improving low-light robustness

3. **Detector choice has more impact than classifier choice** — the 11% fail rate difference between MediaPipe and MMPose has more effect than the 0.5% accuracy difference between MLP and XGBoost

4. **Landmark-based models are fragile under low-light** — 0% robustness without enhancement, because landmarks depend entirely on the detector succeeding

5. **Image-based recognition (YOLO Crop → ResNet18) is inherently more robust to degradation** — the CNN can still extract features from a noisy crop, while landmark coordinates become meaningless when the detector fails

## Naming Convention

Pipeline names follow the pattern:

```
{enhancement}_{representation}_{classifier}
```

Examples:
```
raw_mp_mlp                = No enhancement → MediaPipe → MLP
clahe_mmpose_xgb          = CLAHE → MMPose → XGBoost
zero_dce_yolo_resnet18    = Zero-DCE++ → YOLO Crop → ResNet18
gamma_mp_xgb              = Gamma → MediaPipe → XGBoost
sharpening_mmpose_mlp     = Sharpening → MMPose → MLP
```
