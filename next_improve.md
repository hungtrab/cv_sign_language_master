# Next Improvements

## Current Assessment

The project is good enough for a prepared course demo, but not yet robust enough for real deployment.

The architecture is reasonable:

```text
Webcam frame -> YOLO hand detector -> hand crop -> CNN classifier -> text buffer
```

The main weakness is not the code structure. The main weakness is that the classifier was trained and validated mostly on Kaggle ASL images, while the runtime input is YOLO crops from a real webcam. Those two distributions are different.

## Priority 1: Build a Real Webcam-Crop Dataset

Use the current YOLO hand detector to collect crops from the actual demo pipeline.

Suggested layout:

```text
data/webcam_asl/
  A/
  B/
  C/
  ...
  Z/
  space/
  del/
  nothing/
  unknown/
```

Target amount:

- Minimum quick demo: 50-100 images per important class.
- Better demo: 200-300 images per class.
- Best: multiple people, multiple backgrounds, different lighting, different distances from camera.

Include negative examples:

- no hand
- hand not making a valid sign
- face/body/background crops
- bad detector crops
- partially visible hand

These should go into `nothing` or `unknown`, depending on the final class design.

## Priority 2: Fine-Tune the Classifier

Do not train the classifier from scratch.

Start from the current MobileNetV2 classifier checkpoint and fine-tune with a small learning rate:

```text
lr = 1e-4 or 3e-5
epochs = 5-10
```

Recommended data mix:

```text
70% Kaggle ASL + 30% webcam crops
```

If enough webcam crops are collected, fine-tuning mostly on webcam crops is acceptable.

Useful augmentations:

- small rotation
- small affine transform
- scale/crop jitter
- brightness/contrast jitter
- slight blur
- slight noise
- mild perspective transform

Avoid horizontal flip because many ASL letters are not symmetric.

## Priority 3: Improve Unknown / Rejection Logic

The classifier should not be forced to output a letter for every crop.

Runtime should reject predictions when:

- top-1 confidence is below threshold
- top-1 and top-2 confidence are too close
- detector crop is too large or too small
- prediction is unstable across frames

Current suggested thresholds:

```yaml
classification_conf_threshold: 0.85
classification_margin_threshold: 0.20
```

These should be calibrated on a real webcam validation set, not guessed.

## Priority 4: Prevent Text Spam

The pipeline should emit one letter only when the gesture is stable and the input is armed.

Current desired behavior:

```text
show sign -> stable prediction -> emit one letter
hold same sign -> no repeated spam
release hand / uncertain / nothing -> re-arm
show next sign -> emit next letter
```

The UI should clearly show states:

- `READY`
- `HOLD`
- `ACCEPTED`
- `RELEASE HAND`
- `UNCERTAIN`

This will make the demo easier to use and debug.

## Priority 5: Add Debug Capture

Add a demo key such as `d` to save:

- original frame
- YOLO crop
- predicted label
- confidence
- top-5 predictions
- timestamp

Suggested output:

```text
debug_crops/
  2026-06-04_120001_A_conf092.jpg
  2026-06-04_120001_A_conf092.json
```

This is important because most classifier errors can only be understood by looking at the actual crop given to the classifier.

## Priority 6: Real Evaluation

Do not rely only on Kaggle validation accuracy.

Create a separate webcam test set:

- people not included in training
- different room/background
- different lighting
- each class repeated many times

Report:

- per-class accuracy
- macro-F1
- confusion matrix
- false accept rate for `unknown` / `nothing`
- detector failure rate
- end-to-end latency and FPS

## Priority 7: Decide Demo Scope

For a reliable course demo, it is better to support fewer signs well than all 29 signs badly.

Possible demo subset:

```text
A, B, C, L, O, V, Y, space, del, nothing
```

After the subset works reliably, expand to all classes.

## Not Urgent

These are useful later, but not the first things to fix:

- replacing MobileNetV2 with a larger backbone
- training YOLO again if hand boxes are already acceptable
- adding text-to-speech
- adding dynamic signs like J and Z
- mobile deployment

## Main Conclusion

The next major improvement is data, not architecture.

The detector is acceptable once the correct single-class hand checkpoint is used. The classifier needs real YOLO-cropped webcam data, better rejection logic, and a real validation set before the system can be considered deployable.
