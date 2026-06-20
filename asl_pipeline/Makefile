.PHONY: setup demo demo-raw demo-landmarks demo-enhancement predict evaluate benchmark compare train-classifier train-mlp list-pipelines clean

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

# ── Demos ──

demo:
	python webcam_demo.py --pipeline mediapipe_resnet18

demo-raw:
	python webcam_demo.py --pipeline raw_hf

demo-landmarks:
	python webcam_demo.py --pipeline landmark_mlp

demo-enhancement:
	python webcam_demo.py --pipeline enhancement_clahe_resnet18

# ── Prediction ──

predict:
	python predict_image.py --image $(IMAGE) --pipeline mediapipe_resnet18

# ── Evaluation ──

evaluate:
	python evaluate.py --dataset $(DATASET) --pipeline mediapipe_resnet18

evaluate-all:
	python evaluate.py --dataset $(DATASET) --pipeline mediapipe_resnet18
	python evaluate.py --dataset $(DATASET) --pipeline raw_hf
	python evaluate.py --dataset $(DATASET) --pipeline landmark_mlp
	python evaluate.py --dataset $(DATASET) --pipeline enhancement_clahe_resnet18

benchmark:
	python benchmark_robustness.py --dataset $(DATASET) --pipeline mediapipe_resnet18

benchmark-all:
	python benchmark_robustness.py --dataset $(DATASET) --pipeline mediapipe_resnet18
	python benchmark_robustness.py --dataset $(DATASET) --pipeline raw_hf
	python benchmark_robustness.py --dataset $(DATASET) --pipeline landmark_mlp

compare:
	python compare_pipelines.py --metrics-dir outputs/metrics --output-dir outputs/figures

webcam-eval:
	python evaluate_webcam_recording.py --video $(VIDEO) --label $(LABEL) --pipeline mediapipe_resnet18

# ── Training ──

train-classifier:
	python train_image_classifier.py --dataset $(DATASET) --model resnet18 --epochs 20

train-mlp:
	python train_landmark_mlp.py --features data/landmarks.csv --output weights/landmark_mlp.pkl

# ── Utilities ──

list-pipelines:
	python webcam_demo.py --list-pipelines

clean:
	rm -rf outputs/ debug_crops/ __pycache__ src/asl/__pycache__
