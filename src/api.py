"""API FastAPI para inferência do classificador de notícias."""

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Ajustar o path para poder importar modulos do diretorio src
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "news_classifier.joblib"
DEFAULT_METRICS_PATH = ROOT_DIR / "models" / "news_classifier_metrics.json"


class NewsInput(BaseModel):
    """Payload de entrada para classificação de notícias."""

    titulo: str = Field(..., description="Título da notícia a ser classificado.")
    top_k: Optional[Union[int, str]] = Field(default=3, description="Quantidade de categorias retornadas.")

    @validator("titulo")
    @classmethod
    def validate_titulo(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("O campo 'titulo' deve ser uma string de texto.")

        titulo = value.strip()
        if len(titulo) < 3:
            raise ValueError("O campo 'titulo' precisa ter pelo menos 3 caracteres.")

        return titulo


@dataclass
class ModelServiceConfig:
    """Agrupa os caminhos dos artefatos consumidos pela API."""

    model_path: Path
    metrics_path: Path


class NewsClassifierService:
    """Centraliza carregamento de artefatos e inferência da API."""

    def __init__(self, config: ModelServiceConfig):
        self.config = config
        self._model_cache = None
        self._metrics_cache: dict[str, Any] | None = None

    @staticmethod
    def is_sklearn_pipeline_model(model: Any) -> bool:
        """Identifica se o artefato carregado e um pipeline do scikit-learn."""
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            return False
        if not hasattr(model, "named_steps"):
            return False

        vectorizer = model.named_steps.get("vectorizer")
        classifier = model.named_steps.get("classifier")
        return vectorizer is not None and classifier is not None

    def load_model(self) -> Any | None:
        """Carrega o modelo serializado a partir do disco."""
        if not self.config.model_path.exists():
            return None

        try:
            model = joblib.load(self.config.model_path)
            if not self.is_sklearn_pipeline_model(model):
                print(f"Artefato em formato inválido para a API atual: {self.config.model_path}")
                return None
            print(f"Modelo carregado com sucesso de: {self.config.model_path}")
            return model
        except Exception as exc:
            print(f"Erro ao carregar o modelo de {self.config.model_path}: {exc}")
            return None

    def load_metrics(self) -> dict[str, Any] | None:
        """Carrega as métricas de treino geradas pelo pipeline."""
        if not self.config.metrics_path.exists():
            return None

        try:
            with open(self.config.metrics_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            print(f"Métricas de treino carregadas com sucesso de: {self.config.metrics_path}")
            return data
        except Exception as exc:
            print(f"Erro ao carregar métricas de {self.config.metrics_path}: {exc}")
            return None

    def warm_up(self) -> None:
        """Precarrega caches no startup da aplicação."""
        self._model_cache = self.load_model()
        self._metrics_cache = self.load_metrics()

    def get_model(self) -> Any | None:
        """Retorna o modelo em cache, com fallback de carregamento sob demanda."""
        if self._model_cache is None:
            self._model_cache = self.load_model()
        return self._model_cache

    def get_metrics(self) -> dict[str, Any] | None:
        """Retorna as metricas em cache, com fallback de carregamento sob demanda."""
        if self._metrics_cache is None:
            self._metrics_cache = self.load_metrics()
        return self._metrics_cache

    @staticmethod
    def parse_top_k(value: Optional[Union[int, str]]) -> int:
        """Normaliza o parâmetro top_k, mantendo um padrão seguro em caso de erro."""
        try:
            parsed = int(value) if value is not None else 3
        except (TypeError, ValueError):
            parsed = 3
        return parsed

    @staticmethod
    def get_pipeline_classes(model) -> list[str]:
        """Extrai as classes do pipeline sklearn de forma robusta."""
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")
            classes = getattr(classifier, "classes_", [])

        if hasattr(classes, "tolist"):
            return classes.tolist()
        return list(classes)

    def build_ranked_predictions(self, model, titulo: str) -> list[dict[str, float]]:
        """Executa inferência usando o pipeline do scikit-learn."""
        probabilities = model.predict_proba([titulo])[0]
        classes = self.get_pipeline_classes(model)
        return sorted(
            [
                {"categoria": category, "confianca": round(float(probability), 4)}
                for category, probability in zip(classes, probabilities)
            ],
            key=lambda item: item["confianca"],
            reverse=True,
        )

    @staticmethod
    def build_training_summary(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
        """Resume os metadados principais do último treino."""
        if not metrics:
            return None

        data_summary = metrics.get("data_summary", {})
        return {
            "model_type": metrics.get("model_type"),
            "selection_mode": metrics.get("selection_mode"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "vocabulary_size": metrics.get("vocabulary_size"),
            "classes_ativas": data_summary.get("active_classes", len(metrics.get("classes", []))),
            "amostras_ativas": data_summary.get("active_samples_after_preprocess", metrics.get("total_samples")),
            "amostras_treino": metrics.get("train_samples"),
            "amostras_teste": metrics.get("test_samples"),
            "min_class_frequency": data_summary.get("min_class_frequency"),
            "classes_filtradas": len(data_summary.get("filtered_out_classes", {})),
        }

    def build_health_payload(self) -> dict[str, Any]:
        """Monta a resposta do endpoint de health."""
        model = self.get_model()
        metrics = self.get_metrics()
        summary = self.build_training_summary(metrics)
        return {
            "status": "ok",
            "model_path": str(self.config.model_path),
            "metrics_path": str(self.config.metrics_path),
            "model_loaded": model is not None,
            "metrics_loaded": metrics is not None,
            "selected_model": summary["model_type"] if summary else None,
        }

    def build_metadata_payload(self) -> dict[str, Any]:
        """Monta a resposta detalhada de metadados do modelo."""
        model = self.get_model()
        metrics = self.get_metrics()
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo ainda não foi treinado ou não pode ser carregado.")

        vectorizer = model.named_steps["vectorizer"]
        classifier = model.named_steps["classifier"]
        class_priors = self.build_class_priors_payload(model, classifier)

        return {
            "classes": self.get_pipeline_classes(model),
            "class_priors": class_priors,
            "vocabulary_size": len(getattr(vectorizer, "vocabulary_", {})),
            "alpha": getattr(classifier, "alpha", None),
            "training_summary": self.build_training_summary(metrics),
            "candidate_results": metrics.get("candidate_results") if metrics else None,
            "vectorizer_config": metrics.get("vectorizer_config") if metrics else None,
        }

    def build_class_priors_payload(self, model, classifier) -> dict[str, float]:
        """Converte prioris em probabilidades legíveis quando o classificador expõe esse dado."""
        if not hasattr(classifier, "class_log_prior_"):
            return {}

        class_priors: dict[str, float] = {}
        for category, log_prior in zip(self.get_pipeline_classes(model), classifier.class_log_prior_):
            class_priors[category] = round(math.exp(float(log_prior)), 4)
        return class_priors

    def predict(self, titulo: str, top_k: Optional[Union[int, str]]) -> dict[str, Any]:
        """Realiza a classificação de notícias com base no título recebido."""
        model = self.get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo ainda não foi treinado ou não pode ser carregado.")

        ranked = self.build_ranked_predictions(model, titulo)
        best = ranked[0]
        limited_top_k = max(1, min(self.parse_top_k(top_k), len(ranked)))
        return {
            "categoria_predita": best["categoria"],
            "confianca": best["confianca"],
            "top_categorias": ranked[:limited_top_k],
        }


def build_service_from_environment() -> NewsClassifierService:
    """Constrói o serviço principal a partir das variáveis de ambiente."""
    config = ModelServiceConfig(
        model_path=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        metrics_path=Path(os.getenv("METRICS_PATH", str(DEFAULT_METRICS_PATH))),
    )
    return NewsClassifierService(config)


service = build_service_from_environment()


def get_model() -> Any | None:
    """Wrapper simples para preservar testes e pontos de extensao."""
    return service.get_model()


def get_metrics() -> dict[str, Any] | None:
    """Wrapper simples para preservar testes e pontos de extensao."""
    return service.get_metrics()


app = FastAPI(
    title="AeC News Classifier API",
    version="1.2.0",
    description="API para classificação de notícias em português.",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Inicializa os caches da aplicação no startup."""
    service.warm_up()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Converte erros de validação do FastAPI/Pydantic em respostas 400 legíveis."""
    messages = build_validation_messages(exc)

    detail = " ; ".join(messages) if messages else "Payload deve ser em formato JSON válido."
    return JSONResponse(status_code=400, content={"detail": detail})


def build_validation_messages(exc: RequestValidationError) -> list[str]:
    """Traduz a estrutura de erro do Pydantic para mensagens simples de API."""
    messages: list[str] = []
    for error in exc.errors():
        location = error.get("loc", [])
        field_name = location[-1] if location else "payload"
        message = error.get("msg", "Valor inválido.")
        messages.append(f"{field_name}: {message}")
    return messages


@app.get("/")
def root() -> dict[str, str]:
    """Apresenta um resumo rápido do serviço."""
    return {
        "service": "aec-news-classifier",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metadata": "/metadata",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Endpoint de verificação de integridade da API."""
    return service.build_health_payload()


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Retorna metadados técnicos sobre o modelo atualmente treinado."""
    return service.build_metadata_payload()


@app.post("/predict")
def predict(payload: NewsInput) -> dict[str, Any]:
    """Realiza a classificação de notícias com base no título recebido."""
    return service.predict(payload.titulo, payload.top_k)


def run_server() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"API FastAPI executando em http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
