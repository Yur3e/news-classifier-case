"""Pipeline de treinamento para classificação de títulos de notícias."""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

# Ajustar o path para poder importar modulos do diretorio src
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import limpar_texto

TEXT_COLUMN_CANDIDATES = ["title", "titulo", "headline", "manchete"]
TARGET_COLUMN_CANDIDATES = ["category", "categoria", "classe", "label"]
DEFAULT_MODEL_CANDIDATES = ("logistic_regression", "complement_nb", "naive_bayes")
LARGE_DATASET_AUTO_THRESHOLD = 10
SMALL_DATASET_AUTO_THRESHOLD = 1
AUTO_THRESHOLD_DATASET_SIZE = 1_000


@dataclass
class TrainingConfig:
    """Configura os caminhos e parâmetros principais da execução de treino."""

    dataset_path: Path
    model_output_path: Path
    metrics_output_path: Path
    text_column: str | None = None
    target_column: str | None = None
    max_samples: int | None = None
    model_type: str = "auto"
    min_class_frequency: int | None = None
    test_size: float = 0.2
    seed: int = 42


@dataclass
class TrainingContext:
    """Agrupa o estado do dataset após limpeza e filtros prévios ao treino."""

    texts: list[str]
    labels: list[str]
    raw_samples: int
    raw_class_counts: Counter[str]
    sampled_class_counts: Counter[str]
    removed_class_counts: dict[str, int]
    min_class_frequency: int


class DatasetLoader:
    """Carrega e normaliza o CSV base da classificação."""

    def __init__(self, dataset_path: Path, text_column: str | None, target_column: str | None):
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.target_column = target_column

    def load(self) -> tuple[list[str], list[str]]:
        """Carrega o dataset e extrai as colunas de texto e alvo."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {self.dataset_path}")

        dialect, header = self._read_header()
        text_idx, target_idx = self._resolve_column_indexes(header)
        return self._read_valid_rows(dialect, text_idx, target_idx)

    def _read_header(self) -> tuple[csv.Dialect, list[str]]:
        """Lê o cabeçalho e tenta inferir o dialeto do CSV."""
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            head = file.read(4096)
            file.seek(0)
            try:
                dialect = csv.Sniffer().sniff(head, delimiters=",;|\t")
            except Exception:
                dialect = csv.excel

            reader = csv.reader(file, dialect=dialect)
            header = next(reader)

        return dialect, header

    def _resolve_column_indexes(self, header: list[str]) -> tuple[int, int]:
        """Resolve os índices das colunas de texto e alvo a partir do cabeçalho."""
        header_lower = [col.strip().lower() for col in header]
        text_candidates = [self.text_column.lower()] if self.text_column else TEXT_COLUMN_CANDIDATES
        target_candidates = [self.target_column.lower()] if self.target_column else TARGET_COLUMN_CANDIDATES

        text_idx = self._resolve_column_index(header, header_lower, text_candidates, fallback_index=0, label="texto")
        target_idx = self._resolve_column_index(
            header,
            header_lower,
            target_candidates,
            fallback_index=len(header) - 1 if len(header) > 1 else 0,
            label="alvo",
        )
        return text_idx, target_idx

    def _read_valid_rows(
        self,
        dialect: csv.Dialect,
        text_idx: int,
        target_idx: int,
    ) -> tuple[list[str], list[str]]:
        """Lê apenas linhas com texto e rótulo preenchidos."""
        raw_texts: list[str] = []
        raw_labels: list[str] = []

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, dialect=dialect)
            next(reader)
            for row in reader:
                if len(row) <= max(text_idx, target_idx):
                    continue

                text_val = row[text_idx].strip()
                label_val = row[target_idx].strip().lower()
                if text_val and label_val:
                    raw_texts.append(text_val)
                    raw_labels.append(label_val)

        return raw_texts, raw_labels

    @staticmethod
    def _resolve_column_index(
        header: list[str],
        header_lower: list[str],
        candidates: list[str],
        fallback_index: int,
        label: str,
    ) -> int:
        for candidate in candidates:
            if candidate in header_lower:
                index = header_lower.index(candidate)
                print(f"Coluna de {label} identificada: '{header[index]}'")
                return index

        print(f"Aviso: não detectamos a coluna de {label}. Usando: '{header[fallback_index]}'")
        return fallback_index


class ModelTrainer:
    """Orquestra o pipeline de treino, avaliação e serialização."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def run(self) -> None:
        """Executa o fluxo completo de treinamento."""
        print("=== Iniciando Pipeline de Treinamento ===")
        print(f"Lendo dados de: {self.config.dataset_path}")

        context = self.prepare_training_context()
        train_texts, train_labels, test_texts, test_labels = build_train_test_data(
            context.texts,
            context.labels,
            test_size=self.config.test_size,
            seed=self.config.seed,
        )
        print(f"Dados divididos: Treino={len(train_texts)} | Teste={len(test_texts)}")

        if not train_texts:
            print("Erro: sem dados suficientes para treinamento.")
            sys.exit(1)

        best_pipeline, best_result, candidate_results = train_and_select_model(
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            requested_model_type=self.config.model_type,
        )

        self.save_artifacts(best_pipeline, best_result, candidate_results, context, len(train_texts), len(test_texts))
        self.print_training_summary(best_result)
        print("=== Pipeline de Treinamento Concluído! ===")

    def prepare_training_context(self) -> TrainingContext:
        """Prepara e limpa o dataset antes do treino."""
        loader = DatasetLoader(
            dataset_path=self.config.dataset_path,
            text_column=self.config.text_column,
            target_column=self.config.target_column,
        )

        texts, labels = self._load_raw_dataset(loader)

        raw_samples = len(texts)
        raw_class_counts = Counter(labels)
        print(f"Total de linhas válidas lidas: {raw_samples}")

        texts, labels = sample_dataset(texts, labels, self.config.max_samples)
        sampled_class_counts = Counter(labels)

        print("Aplicando limpeza de texto e removendo entradas vazias...")
        filtered_texts, filtered_labels = filter_empty_after_preprocess(texts, labels)
        print(f"Dataset limpo e ativo contém: {len(filtered_texts)} registros com tokens válidos.")

        if not filtered_texts:
            print("Erro: sem textos válidos após a limpeza.")
            sys.exit(1)

        min_class_frequency = resolve_min_class_frequency(self.config.min_class_frequency, len(filtered_texts))
        print(f"Limiar de classes raras adotado: {min_class_frequency}")
        filtered_texts, filtered_labels, removed_class_counts = filter_classes_by_frequency(
            filtered_texts,
            filtered_labels,
            min_class_frequency=min_class_frequency,
        )

        self._print_rare_class_summary(removed_class_counts)

        if not filtered_texts:
            print("Erro: todas as amostras foram removidas após o filtro de frequência.")
            sys.exit(1)

        return TrainingContext(
            texts=filtered_texts,
            labels=filtered_labels,
            raw_samples=raw_samples,
            raw_class_counts=raw_class_counts,
            sampled_class_counts=sampled_class_counts,
            removed_class_counts=removed_class_counts,
            min_class_frequency=min_class_frequency,
        )

    @staticmethod
    def _load_raw_dataset(loader: DatasetLoader) -> tuple[list[str], list[str]]:
        """Executa o carregamento do CSV e encerra a execução com mensagem amigável em caso de erro."""
        try:
            return loader.load()
        except Exception as exc:
            print(f"Erro ao carregar dados: {exc}")
            sys.exit(1)

    @staticmethod
    def _print_rare_class_summary(removed_class_counts: dict[str, int]) -> None:
        """Resume no console o impacto do filtro de frequência mínima por classe."""
        if removed_class_counts:
            print(
                f"Classes raras removidas: {len(removed_class_counts)} "
                f"(amostras descartadas: {sum(removed_class_counts.values())})"
            )
            return

        print("Nenhuma classe rara foi removida nesta execução.")

    def save_artifacts(
        self,
        best_pipeline: Pipeline,
        best_result: dict[str, Any],
        candidate_results: list[dict[str, Any]],
        context: TrainingContext,
        train_samples: int,
        test_samples: int,
    ) -> None:
        """Persistem modelo e metadados do experimento em disco."""
        print("Salvando modelo selecionado...")
        self.config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, self.config.model_output_path)
        print(f"Modelo salvo em: {self.config.model_output_path}")

        metrics_data = self._build_metrics_payload(
            best_pipeline,
            best_result,
            candidate_results,
            context,
            train_samples,
            test_samples,
        )

        print(f"Salvando métricas em: {self.config.metrics_output_path}")
        self.config.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_output_path, "w", encoding="utf-8") as file:
            json.dump(metrics_data, file, ensure_ascii=False, indent=2)

    def _build_metrics_payload(
        self,
        best_pipeline: Pipeline,
        best_result: dict[str, Any],
        candidate_results: list[dict[str, Any]],
        context: TrainingContext,
        train_samples: int,
        test_samples: int,
    ) -> dict[str, Any]:
        """Monta o JSON final de métricas e metadados do experimento."""
        vectorizer = best_pipeline.named_steps["vectorizer"]
        classifier = best_pipeline.named_steps["classifier"]

        return {
            "model_type": best_result["model_type"],
            "selection_mode": self.config.model_type,
            "selection_metric": "macro_f1",
            "total_samples": len(context.texts),
            "train_samples": train_samples,
            "test_samples": test_samples,
            "accuracy": round(best_result["accuracy"], 4) if "accuracy" in best_result else None,
            "macro_f1": round(best_result["macro_f1"], 4) if "macro_f1" in best_result else None,
            "weighted_f1": round(best_result["weighted_f1"], 4) if "weighted_f1" in best_result else None,
            "classification_report": round_float_metrics(best_result.get("classification_report")),
            "classification_report_text": best_result.get("classification_report_text"),
            "candidate_results": round_float_metrics(candidate_results),
            "classes": classifier.classes_.tolist(),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "vectorizer_config": {
                "ngram_range": list(getattr(vectorizer, "ngram_range", (1, 1))),
                "min_df": getattr(vectorizer, "min_df", None),
                "max_df": getattr(vectorizer, "max_df", None),
                "sublinear_tf": getattr(vectorizer, "sublinear_tf", None),
            },
            "data_summary": self._build_data_summary(context),
            "training_config": round_float_metrics(asdict(self.config)),
        }

    def _build_data_summary(self, context: TrainingContext) -> dict[str, Any]:
        """Resume o recorte do dataset que realmente chegou ao treino."""
        return {
            "dataset_path": str(self.config.dataset_path),
            "raw_samples": context.raw_samples,
            "sampled_samples": sum(context.sampled_class_counts.values()),
            "active_samples_after_preprocess": len(context.texts),
            "raw_classes": len(context.raw_class_counts),
            "sampled_classes": len(context.sampled_class_counts),
            "active_classes": len(set(context.labels)),
            "min_class_frequency": context.min_class_frequency,
            "filtered_out_samples": sum(context.removed_class_counts.values()),
            "filtered_out_classes": context.removed_class_counts,
            "top_active_classes": summarize_top_classes(context.labels),
        }

    @staticmethod
    def print_training_summary(best_result: dict[str, Any]) -> None:
        """Exibe no console o resumo da melhor rodada de treino."""
        if "classification_report_text" not in best_result:
            return

        print("\nResultado da avaliação do modelo selecionado:")
        print(f"Accuracy Global: {best_result['accuracy'] * 100:.2f}%")
        print(f"Macro F1: {best_result['macro_f1']:.4f}")
        print(f"Weighted F1: {best_result['weighted_f1']:.4f}")
        print("\nClassification Report:")
        print(best_result["classification_report_text"])


def sample_dataset(texts: list[str], labels: list[str], max_samples: int | None) -> tuple[list[str], list[str]]:
    """Limita a base mantendo aleatoriedade reprodutível quando `max_samples` é informado."""
    if not max_samples or len(texts) <= max_samples:
        return texts, labels

    print(f"Amostrando dataset para o limite de max-samples: {max_samples}")
    random_gen = random.Random(42)
    indices = random_gen.sample(range(len(texts)), max_samples)
    sampled_texts = [texts[index] for index in indices]
    sampled_labels = [labels[index] for index in indices]
    return sampled_texts, sampled_labels


def filter_empty_after_preprocess(texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    """Remove linhas cujo texto fica vazio após a limpeza/tokenização."""
    filtered_texts: list[str] = []
    filtered_labels: list[str] = []

    for idx, (text, label) in enumerate(zip(texts, labels)):
        if idx > 0 and idx % 20_000 == 0:
            print(f"  > Processadas {idx} linhas...")

        if limpar_texto(text):
            filtered_texts.append(text)
            filtered_labels.append(label)

    return filtered_texts, filtered_labels


def resolve_min_class_frequency(requested_min_frequency: int | None, dataset_size: int) -> int:
    """Resolve automaticamente um limiar de corte para classes raras."""
    if requested_min_frequency is not None:
        return max(1, requested_min_frequency)
    if dataset_size >= AUTO_THRESHOLD_DATASET_SIZE:
        return LARGE_DATASET_AUTO_THRESHOLD
    return SMALL_DATASET_AUTO_THRESHOLD


def filter_classes_by_frequency(
    texts: list[str],
    labels: list[str],
    min_class_frequency: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Remove classes raras demais para aprendizado supervisionado confiável."""
    if min_class_frequency <= 1:
        return texts, labels, {}

    class_counts = Counter(labels)
    allowed_labels = {label for label, count in class_counts.items() if count >= min_class_frequency}

    filtered_texts: list[str] = []
    filtered_labels: list[str] = []
    removed_class_counts: Counter[str] = Counter()

    for text, label in zip(texts, labels):
        if label in allowed_labels:
            filtered_texts.append(text)
            filtered_labels.append(label)
        else:
            removed_class_counts[label] += 1

    return filtered_texts, filtered_labels, dict(sorted(removed_class_counts.items()))


def build_train_test_data(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Divide treino/teste com estratificação quando possível e preserva classes raras no treino."""
    class_counts = Counter(labels)
    forced_train_indices = [idx for idx, label in enumerate(labels) if class_counts[label] < 2]
    split_indices = [idx for idx, label in enumerate(labels) if class_counts[label] >= 2]

    train_texts = [texts[idx] for idx in forced_train_indices]
    train_labels = [labels[idx] for idx in forced_train_indices]
    test_texts: list[str] = []
    test_labels: list[str] = []

    split_texts = [texts[idx] for idx in split_indices]
    split_labels = [labels[idx] for idx in split_indices]

    if len(split_texts) < 2:
        return train_texts + split_texts, train_labels + split_labels, test_texts, test_labels

    # A estratificação é preferível, mas não é possível em alguns recortes muito pequenos.
    stratify_labels = split_labels if len(set(split_labels)) > 1 else None

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            split_texts,
            split_labels,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            split_texts,
            split_labels,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    return train_texts + x_train, train_labels + y_train, x_test, y_test


def build_vectorizer(total_samples: int) -> TfidfVectorizer:
    """Cria um vetorizador TF-IDF mais robusto para classificação de títulos."""
    is_large_dataset = total_samples >= AUTO_THRESHOLD_DATASET_SIZE
    return TfidfVectorizer(
        tokenizer=limpar_texto,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2 if is_large_dataset else 1,
        max_df=0.98 if is_large_dataset else 1.0,
        sublinear_tf=True,
    )


def build_pipeline(model_type: str, total_samples: int) -> Pipeline:
    """Cria o pipeline de vetorização TF-IDF e classificação."""
    classifier = build_classifier(model_type)

    return Pipeline(
        steps=[
            ("vectorizer", build_vectorizer(total_samples)),
            ("classifier", classifier),
        ]
    )


def build_classifier(model_type: str) -> LogisticRegression | ComplementNB | MultinomialNB:
    """Cria apenas o classificador, separado do vetorizador para simplificar leitura e testes."""
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=3.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        )
    if model_type == "complement_nb":
        return ComplementNB(alpha=0.5)
    return MultinomialNB(alpha=0.5)


def round_float_metrics(data: Any) -> Any:
    """Arredonda recursivamente métricas numéricas para facilitar leitura do JSON."""
    if isinstance(data, dict):
        return {key: round_float_metrics(value) for key, value in data.items()}
    if isinstance(data, list):
        return [round_float_metrics(value) for value in data]
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, float):
        return round(data, 4)
    return data


def evaluate_pipeline(pipeline: Pipeline, test_texts: list[str], test_labels: list[str]) -> dict[str, Any]:
    """Avalia um pipeline treinado no holdout e retorna métricas prontas para serialização."""
    predictions = pipeline.predict(test_texts)
    accuracy = float(accuracy_score(test_labels, predictions))
    macro_f1 = float(f1_score(test_labels, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(test_labels, predictions, average="weighted", zero_division=0))
    report_dict = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
    report_text = classification_report(test_labels, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def build_candidate_models(requested_model_type: str) -> list[str]:
    """Resolve a lista de modelos candidatos para a rodada de treino."""
    if requested_model_type == "auto":
        return list(DEFAULT_MODEL_CANDIDATES)
    return [requested_model_type]


def select_best_candidate(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Seleciona o melhor candidato priorizando macro F1, depois weighted F1 e accuracy."""
    return max(
        candidate_results,
        key=lambda result: (
            result.get("macro_f1", float("-inf")),
            result.get("weighted_f1", float("-inf")),
            result.get("accuracy", float("-inf")),
        ),
    )


def train_and_select_model(
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    requested_model_type: str,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Treina os candidatos e retorna o melhor pipeline."""
    candidate_results: list[dict[str, Any]] = []
    trained_pipelines: dict[str, Pipeline] = {}
    candidate_models = build_candidate_models(requested_model_type)

    print(f"Modelos candidatos: {', '.join(candidate_models)}")

    for model_type in candidate_models:
        print(f"Treinando pipeline TfidfVectorizer + {model_type}...")
        pipeline = build_pipeline(model_type, len(train_texts))
        pipeline.fit(train_texts, train_labels)
        trained_pipelines[model_type] = pipeline

        result: dict[str, Any] = {"model_type": model_type}
        if test_texts:
            result.update(evaluate_pipeline(pipeline, test_texts, test_labels))
            print(
                f"  -> accuracy={result['accuracy']:.4f} | "
                f"macro_f1={result['macro_f1']:.4f} | weighted_f1={result['weighted_f1']:.4f}"
            )
        else:
            print("  -> sem holdout suficiente; mantendo candidato disponível sem comparação.")

        candidate_results.append(result)

    best_result = select_best_candidate(candidate_results)
    best_pipeline = trained_pipelines[best_result["model_type"]]
    print(f"Modelo selecionado: {best_result['model_type']}")
    return best_pipeline, best_result, candidate_results


def summarize_top_classes(labels: list[str], limit: int = 15) -> list[dict[str, int]]:
    """Resume as classes mais frequentes do conjunto ativo."""
    return [
        {"categoria": label, "amostras": count}
        for label, count in Counter(labels).most_common(limit)
    ]


def build_training_config_from_args() -> TrainingConfig:
    """Traduz os argumentos de CLI para uma configuração tipada."""
    parser = argparse.ArgumentParser(description="Treinamento CLI de Classificador de Notícias com scikit-learn")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/articles.csv",
        help="Caminho do arquivo de dados (CSV)",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/news_classifier.joblib",
        help="Saída do pipeline final treinado",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="models/news_classifier_metrics.json",
        help="Saída do JSON de métricas",
    )
    parser.add_argument("--text-column", type=str, default=None, help="Nome da coluna de texto")
    parser.add_argument("--target-column", type=str, default=None, help="Nome da coluna alvo")
    parser.add_argument("--max-samples", type=int, default=None, help="Número máximo de amostras para ler")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "naive_bayes", "complement_nb", "logistic_regression"],
        default="auto",
        help="Algoritmo de classificação ou seleção automática entre candidatos",
    )
    parser.add_argument(
        "--min-class-frequency",
        type=int,
        default=None,
        help="Remove classes com menos exemplos do que o limiar informado. Em auto: 10 para bases grandes, 1 para bases pequenas.",
    )
    args = parser.parse_args()

    return TrainingConfig(
        dataset_path=Path(args.dataset),
        model_output_path=Path(args.model_output),
        metrics_output_path=Path(args.metrics_output),
        text_column=args.text_column,
        target_column=args.target_column,
        max_samples=args.max_samples,
        model_type=args.model_type,
        min_class_frequency=args.min_class_frequency,
    )


def main() -> None:
    trainer = ModelTrainer(build_training_config_from_args())
    trainer.run()


if __name__ == "__main__":
    main()
