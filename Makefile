.PHONY: setup data train-detector train-classifier evaluate demo demo-web test lint clean

PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

DET_CONFIG ?= configs/detector.yaml
CLS_CONFIG ?= configs/classifier.yaml
DEMO_CONFIG ?= configs/demo.yaml

setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

data:
	@mkdir -p data/asl_alphabet data/hagrid_hand data/yolo
	@cat docs/DATA.md | head -40
	$(PY) -m scripts.prepare_yolo_data --help

train-detector: setup
	$(PY) -m scripts.train_detector --config $(DET_CONFIG)

train-classifier: setup
	$(PY) -m scripts.train_classifier --config $(CLS_CONFIG)

evaluate: setup
	$(PY) -m scripts.evaluate_pipeline --config $(DEMO_CONFIG)

demo: setup
	$(PY) -m scripts.run_demo --config $(DEMO_CONFIG) --backend cv2

demo-web: setup
	$(PY) -m scripts.run_demo --config $(DEMO_CONFIG) --backend gradio

test: setup
	$(PY) -m pytest

lint: setup
	$(VENV)/bin/ruff check src tests

clean:
	rm -rf $(VENV) **/__pycache__ .pytest_cache .ruff_cache runs/
