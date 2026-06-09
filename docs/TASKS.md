# Phân chia công việc — Nhóm 3 người (CV)

| # | Người | MSSV | Mảng | Files chính |
|---|---|---|---|---|
| 1 | **Trần Quang Hưng** | 20235502 | YOLO + dataset prep | `src/signlang/detection/`, `src/signlang/data/prepare_yolo_data.py`, `scripts/train_detector.py`, `configs/detector.yaml` |
| 2 | **Đỗ Đăng Vũ** | 20235578 | Classifier + transforms | `src/signlang/classification/`, `src/signlang/data/{asl_dataset,transforms}.py`, `scripts/train_classifier.py`, `configs/classifier.yaml` |
| 3 | **Lê Hoàng Tùng** | 20235572 | Pipeline + demo + eval + report | `src/signlang/{pipeline,demo,evaluation,utils}/`, `scripts/{run_demo,evaluate_pipeline}.py`, `tests/`, slide+báo cáo |

## Acceptance per role

### Hưng (detection)
- HaGRID subset converted -> `data/yolo/dataset.yaml`.
- `make train-detector` runs on Colab GPU (1 epoch sanity).
- `runs/detector/weights/best.pt` exists; `HandDetector(weights).detect(frame)` returns `Detection(...)` with sane bbox.

### Vũ (classifier)
- `make train-classifier CLS_CONFIG=configs/classifier.yaml` produces `runs/classifier_resnet18/best.pt`.
- Both `arch=resnet18` and `arch=mobilenet_v2` train without code changes (test_config.py covers).
- `LetterClassifier(weights).predict(rgb)` returns `Prediction(label, confidence, top5)`.

### Tùng (pipeline + demo)
- `python -m scripts.run_demo --config configs/demo.yaml` opens an OpenCV window with bbox + predicted letter.
- `--backend gradio` opens a browser demo.
- `make evaluate` reports detection rate, accuracy, macro-F1 + saves confusion matrix CSV.

## Mốc thời gian gợi ý

| Tuần | Việc |
|---|---|
| 1 | Mỗi người scaffold module của mình; CSV tests pass; data folder structure chuẩn |
| 2 | Hưng có YOLO checkpoint baseline; Vũ có classifier baseline (~85% val acc) |
| 3 | Tích hợp pipeline (Tùng); fine-tune detector + classifier để đạt mục tiêu (>90% mAP / >95% acc) |
| 4 | Demo polish (UI, smoothing, save/clear); đo FPS; báo cáo + slide |

## Báo cáo (10 trang)

| Mục | Người |
|---|---|
| 1. Introduction & motivation | cả nhóm |
| 2. Related work | Tùng |
| 3. Detection (YOLOv8) | Hưng |
| 4. Classification (ResNet/MN-v2 + ablation) | Vũ |
| 5. System integration | Tùng |
| 6. Experiments & results | cả nhóm |
| 7. Demo + UX | Tùng |
| 8. Conclusion | cả nhóm |
