from pathlib import Path

import pytest

from signlang.utils.config import Config, load_class_names, load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_config_attribute_access():
    cfg = load_config(CONFIG_DIR / "classifier.yaml")
    assert cfg.experiment_name == "cls_resnet18"
    assert cfg.model.arch == "resnet18"
    assert cfg.model.num_classes == 29
    assert cfg.data.image_size == 224


def test_class_names_count_matches_classifier():
    names = load_class_names(CONFIG_DIR / "asl_classes.txt")
    assert len(names) == 29
    assert names[0] == "A"
    assert names[-1] == "nothing"


def test_classifier_factory_resnet18():
    pytest.importorskip("torchvision")
    from signlang.classification.models import build_classifier, count_parameters
    cfg = Config({
        "model": {"arch": "resnet18", "num_classes": 29,
                  "pretrained": False, "freeze_backbone": False},
    })
    model = build_classifier(cfg)
    trainable, total = count_parameters(model)
    assert trainable == total
    # last layer should output 29
    out = model(__import__("torch").zeros(1, 3, 224, 224))
    assert out.shape == (1, 29)


def test_classifier_factory_mobilenet():
    pytest.importorskip("torchvision")
    from signlang.classification.models import build_classifier
    cfg = Config({
        "model": {"arch": "mobilenet_v2", "num_classes": 29,
                  "pretrained": False, "freeze_backbone": False},
    })
    model = build_classifier(cfg)
    out = model(__import__("torch").zeros(1, 3, 224, 224))
    assert out.shape == (1, 29)
