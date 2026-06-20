from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .logging import get_logger

log = get_logger("asl.download")

MODELS = [
    {
        "name": "resnet18_huzaifanasirrr",
        "repo_id": "huzaifanasirrr/realtime-sign-language-translator",
        "files": ["best_model.pth", "class_mapping.json"],
        "description": "ResNet18 ASL classifier (26 classes)",
    },
    {
        "name": "siglip_prithiv",
        "repo_id": "prithivMLmods/Alphabet-Sign-Language-Detection",
        "files": [],
        "description": "SigLIP-based ASL classifier (HF Transformers)",
    },
    {
        "name": "resnet50_abuzaid",
        "repo_id": "Abuzaid01/asl-sign-language-classifier",
        "files": [],
        "description": "ResNet50 ASL classifier (29 classes, filter to A-Z)",
    },
]


def download_hf_file(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> Path:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    return Path(path)


def download_model_option1(cache_dir: Optional[str] = None) -> dict:
    repo = MODELS[0]["repo_id"]
    log.info(f"Trying Option 1: {repo}")
    weights_path = download_hf_file(repo, "best_model.pth", cache_dir)
    mapping_path = download_hf_file(repo, "class_mapping.json", cache_dir)
    with open(mapping_path) as f:
        class_mapping = json.load(f)
    return {
        "source": "resnet18_huzaifanasirrr",
        "weights_path": weights_path,
        "class_mapping": class_mapping,
        "arch": "resnet18",
    }


def download_model_option2(cache_dir: Optional[str] = None) -> dict:
    repo = MODELS[1]["repo_id"]
    log.info(f"Trying Option 2: {repo}")
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    try:
        processor = AutoImageProcessor.from_pretrained(repo, cache_dir=cache_dir)
        model = AutoModelForImageClassification.from_pretrained(repo, cache_dir=cache_dir)
    except Exception:
        from transformers import SiglipForImageClassification
        processor = AutoImageProcessor.from_pretrained(repo, cache_dir=cache_dir)
        model = SiglipForImageClassification.from_pretrained(repo, cache_dir=cache_dir)
    return {
        "source": "siglip_prithiv",
        "processor": processor,
        "model": model,
        "arch": "hf_transformers",
    }


def download_model_option3(cache_dir: Optional[str] = None) -> dict:
    repo = MODELS[2]["repo_id"]
    log.info(f"Trying Option 3: {repo}")
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained(repo, cache_dir=cache_dir)
    model = AutoModelForImageClassification.from_pretrained(repo, cache_dir=cache_dir)
    return {
        "source": "resnet50_abuzaid",
        "processor": processor,
        "model": model,
        "arch": "hf_transformers",
    }


def load_recognizer_with_fallback(
    force_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    options = [
        ("Option 1", download_model_option1),
        ("Option 2", download_model_option2),
        ("Option 3", download_model_option3),
    ]

    if force_model:
        for name, fn in options:
            if force_model.lower() in name.lower():
                return fn(cache_dir)
        raise ValueError(f"Unknown forced model: {force_model}")

    for name, fn in options:
        try:
            result = fn(cache_dir)
            log.info(f"{name} loaded successfully: {result['source']}")
            return result
        except Exception as e:
            log.warning(f"{name} failed: {e}")

    raise RuntimeError(
        "All model options failed. Check network connection and installed packages."
    )
