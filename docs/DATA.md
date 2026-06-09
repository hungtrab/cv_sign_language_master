# Datasets

Two datasets are needed: one for hand detection, one for letter
classification.

## 1. ASL Alphabet (classification)

Kaggle: <https://www.kaggle.com/datasets/grassknoted/asl-alphabet>

```
data/asl_alphabet/
├── asl_alphabet_train/
│   ├── A/    *.jpg     (~3000 images)
│   ├── B/    *.jpg
│   ├── ...
│   ├── Z/
│   ├── space/
│   ├── del/
│   └── nothing/
└── asl_alphabet_test/  (one image per class)
```

87,000 training images, 200×200 RGB. Folder names must match
`configs/asl_classes.txt` (uppercase letters + `space`, `del`,
`nothing`).

## 2. HaGRID-derived hand detection

We use a subset of HaGRID (any "hand visible" gesture) to fine-tune
YOLOv8 on a single class — `hand`.

Recommended steps:

```bash
# Download a manageable subset (~5000 images)
python scripts/download_hagrid_subset.py --target data/hagrid_hand --n 5000

# Convert annotations to YOLOv8 format
python -m scripts.prepare_yolo_data \
    --images data/hagrid_hand/images \
    --annotations data/hagrid_hand/annotations.json \
    --output data/yolo
```

(The download script is intentionally not shipped — HaGRID's full
release is multi-gigabyte; the team can swap in any hand-bbox dataset
with a tiny adapter.)

The output `data/yolo/dataset.yaml` is what `configs/detector.yaml`
points to.

## 3. Sign Language MNIST (optional baseline)

<https://www.kaggle.com/datamunge/sign-language-mnist> — used for fast
prototyping in the v1 project. Not required for the YOLO+CNN pipeline.
