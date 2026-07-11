import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import pytest
from fastapi.testclient import TestClient

# Ajustar o path para poder importar modulos do diretorio src
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import src.api as api
from src.train import build_pipeline


class MockVectorizer:
    vocabulary_ = {"futebol": 0, "eleicao": 1, "presidente": 2}


class MockClassifier:
    classes_ = ["esporte", "politica"]
    class_log_prior_ = [math.log(0.5), math.log(0.5)]
    alpha = 1.0


class MockPipeline:
    def __init__(self):
        self.classes_ = ["esporte", "politica"]
        self.named_steps = {
            "vectorizer": MockVectorizer(),
            "classifier": MockClassifier(),
        }

    def predict(self, texts):
        return [self._predict_category(text) for text in texts]

    def predict_proba(self, texts):
        probabilities = []
        for text in texts:
            category = self._predict_category(text)
            if category == "esporte":
                probabilities.append([0.9, 0.1])
            else:
                probabilities.append([0.1, 0.9])
        return probabilities

    @staticmethod
    def _predict_category(text: str) -> str:
        normalized = text.lower()
        if "futebol" in normalized:
            return "esporte"
        return "politica"


MOCK_MODEL = MockPipeline()


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def trained_client(tmp_path, monkeypatch):
    texts = [
        "Time vence classico no campeonato nacional",
        "Atacante marca tres gols em partida decisiva",
        "Tecnico ajusta elenco para semifinal do torneio",
        "Presidente negocia apoio no congresso",
        "Senado aprova novo projeto de lei",
        "Ministro discute reforma tributaria em brasilia",
    ]
    labels = ["esporte", "esporte", "esporte", "politica", "politica", "politica"]

    pipeline = build_pipeline("naive_bayes", len(texts))
    pipeline.fit(texts, labels)

    model_path = tmp_path / "news_classifier.joblib"
    metrics_path = tmp_path / "news_classifier_metrics.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "model_type": "naive_bayes",
                "selection_mode": "auto",
                "accuracy": 0.95,
                "macro_f1": 0.95,
                "weighted_f1": 0.95,
                "classes": ["esporte", "politica"],
                "vocabulary_size": 32,
                "candidate_results": [
                    {"model_type": "naive_bayes", "accuracy": 0.95, "macro_f1": 0.95, "weighted_f1": 0.95}
                ],
                "vectorizer_config": {"ngram_range": [1, 2], "min_df": 1, "max_df": 1.0, "sublinear_tf": True},
                "data_summary": {
                    "active_classes": 2,
                    "active_samples_after_preprocess": 6,
                    "min_class_frequency": 1,
                    "filtered_out_classes": {},
                },
                "train_samples": 4,
                "test_samples": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(api.service.config, "model_path", model_path)
    monkeypatch.setattr(api.service.config, "metrics_path", metrics_path)
    monkeypatch.setattr(api.service, "_model_cache", None)
    monkeypatch.setattr(api.service, "_metrics_cache", None)

    with TestClient(api.app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "aec-news-classifier"
    assert data["predict"] == "/predict"


@patch("src.api.service.get_model")
@patch("src.api.service.get_metrics")
def test_health_endpoint(mock_get_metrics, mock_get_model, client):
    mock_get_model.return_value = MOCK_MODEL
    mock_get_metrics.return_value = {"model_type": "naive_bayes", "data_summary": {"active_classes": 2}}

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["metrics_loaded"] is True


@patch("src.api.service.get_model")
def test_metadata_endpoint_unloaded(mock_get_model, client):
    mock_get_model.return_value = None
    response = client.get("/metadata")
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data


@patch("src.api.service.get_model")
@patch("src.api.service.get_metrics")
def test_metadata_endpoint_loaded(mock_get_metrics, mock_get_model, client):
    mock_get_model.return_value = MOCK_MODEL
    mock_get_metrics.return_value = {
        "model_type": "naive_bayes",
        "accuracy": 0.88,
        "macro_f1": 0.84,
        "weighted_f1": 0.87,
        "candidate_results": [{"model_type": "naive_bayes", "accuracy": 0.88, "macro_f1": 0.84, "weighted_f1": 0.87}],
        "vectorizer_config": {"ngram_range": [1, 2], "min_df": 1, "max_df": 1.0, "sublinear_tf": True},
        "data_summary": {
            "active_classes": 2,
            "active_samples_after_preprocess": 120,
            "min_class_frequency": 1,
            "filtered_out_classes": {},
        },
        "train_samples": 96,
        "test_samples": 24,
    }

    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["classes"] == ["esporte", "politica"]
    assert "class_priors" in data
    assert data["vocabulary_size"] == 3
    assert data["training_summary"]["model_type"] == "naive_bayes"
    assert data["candidate_results"][0]["macro_f1"] == 0.84


@patch("src.api.service.get_model")
def test_predict_endpoint_success(mock_get_model, client):
    mock_get_model.return_value = MOCK_MODEL

    payload = {"titulo": "Futebol no final de semana foi incrivel", "top_k": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["categoria_predita"] == "esporte"
    assert "confianca" in data
    assert len(data["top_categorias"]) <= 2

    payload = {"titulo": "Eleicao presidencial movimentou o pais", "top_k": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["categoria_predita"] == "politica"
    assert len(data["top_categorias"]) == 1


@patch("src.api.service.get_model")
def test_predict_endpoint_validations(mock_get_model, client):
    mock_get_model.return_value = MOCK_MODEL

    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "titulo" in response.json()["detail"]

    response = client.post("/predict", json={"titulo": "oi"})
    assert response.status_code == 400
    assert "caracteres" in response.json()["detail"]

    response = client.post("/predict", data="texto puro invalido")
    assert response.status_code == 400


def test_predict_endpoint_with_real_pipeline(trained_client):
    payload = {"titulo": "Atacante decide a final do campeonato", "top_k": "2"}
    response = trained_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["categoria_predita"] == "esporte"
    assert len(data["top_categorias"]) == 2


def test_metadata_endpoint_with_real_artifacts(trained_client):
    response = trained_client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["training_summary"]["accuracy"] == 0.95
    assert data["training_summary"]["classes_ativas"] == 2
    assert data["vectorizer_config"]["ngram_range"] == [1, 2]
