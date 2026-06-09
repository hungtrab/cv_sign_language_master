import numpy as np

from signlang.evaluation import accuracy, confusion_matrix, macro_f1, top_k_accuracy


def test_accuracy_perfect():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_zero():
    assert accuracy([0, 1, 2], [2, 0, 1]) == 0.0


def test_top_k_accuracy_k_equals_classes():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((10, 5))
    labels = rng.integers(0, 5, size=10)
    assert top_k_accuracy(logits, labels, k=5) == 1.0


def test_macro_f1_perfect():
    f1 = macro_f1([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], num_classes=3)
    assert f1 == 1.0


def test_confusion_matrix_shape():
    cm = confusion_matrix([0, 1, 1, 2], [0, 1, 0, 2], num_classes=3)
    assert cm.shape == (3, 3)
    assert cm.sum() == 4
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm[0, 1] == 1   # one ground-truth-0 predicted as 1
