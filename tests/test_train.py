import sys
from pathlib import Path

# Ajustar o path para poder importar modulos do diretorio src
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.train import (
    build_candidate_models,
    filter_classes_by_frequency,
    resolve_min_class_frequency,
    select_best_candidate,
)


def test_resolve_min_class_frequency_auto_for_small_dataset():
    assert resolve_min_class_frequency(None, dataset_size=100) == 1


def test_resolve_min_class_frequency_auto_for_large_dataset():
    assert resolve_min_class_frequency(None, dataset_size=5_000) == 10


def test_filter_classes_by_frequency_removes_only_rare_labels():
    texts = ["a", "b", "c", "d", "e", "f"]
    labels = ["esporte", "esporte", "politica", "politica", "rarissima", "outra"]

    filtered_texts, filtered_labels, removed = filter_classes_by_frequency(texts, labels, min_class_frequency=2)

    assert filtered_texts == ["a", "b", "c", "d"]
    assert filtered_labels == ["esporte", "esporte", "politica", "politica"]
    assert removed == {"outra": 1, "rarissima": 1}


def test_build_candidate_models_auto_returns_all_supported_candidates():
    assert build_candidate_models("auto") == ["logistic_regression", "complement_nb", "naive_bayes"]


def test_select_best_candidate_prioritizes_macro_f1():
    candidates = [
        {"model_type": "naive_bayes", "accuracy": 0.92, "macro_f1": 0.71, "weighted_f1": 0.82},
        {"model_type": "logistic_regression", "accuracy": 0.90, "macro_f1": 0.75, "weighted_f1": 0.80},
        {"model_type": "complement_nb", "accuracy": 0.93, "macro_f1": 0.75, "weighted_f1": 0.83},
    ]

    best = select_best_candidate(candidates)

    assert best["model_type"] == "complement_nb"
