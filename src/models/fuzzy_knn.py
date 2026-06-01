from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.config import DEFAULT_RANDOM_SEED
from src.preprocessing import prepare_dataset_splits


class FuzzyKNNClassifier(BaseEstimator, ClassifierMixin):
    # Classificador Fuzzy KNN com pesos por distancia e saida probabilistica.

    def __init__(
        self,
        n_neighbors: int = 5,
        m: float = 2.0,
        metric: str = "euclidean",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "FuzzyKNNClassifier":
        # Armazena o conjunto de treino e prepara a estrutura de vizinhos.
        X_validated, y_validated = check_X_y(X, y, dtype=float)

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors deve ser maior ou igual a 1.")
        if self.n_neighbors > X_validated.shape[0]:
            raise ValueError(
                "n_neighbors nao pode ser maior que o numero de amostras de treino."
            )
        if self.m <= 1.0:
            raise ValueError("m deve ser maior que 1.0 no Fuzzy KNN.")

        self.classes_ = np.unique(y_validated)
        self.class_to_index_ = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.X_train_ = X_validated
        self.y_train_ = y_validated
        self.n_features_in_ = X_validated.shape[1]
        self.training_membership_ = self._build_training_membership(y_validated)
        self.neighbor_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )
        self.neighbor_model_.fit(X_validated)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Prediz a classe mais provavel segundo as pertinencias calculadas.
        class_probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(class_probabilities, axis=1)
        return self.classes_[predicted_indices]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Calcula pertinencias de classe a partir dos k vizinhos mais proximos.
        check_is_fitted(
            self,
            ["classes_", "training_membership_", "neighbor_model_", "X_train_"],
        )
        X_validated = check_array(X, dtype=float)
        if X_validated.shape[1] != self.n_features_in_:
            raise ValueError(
                "Numero de atributos inconsistente com o conjunto de treino."
            )

        distances, neighbor_indices = self.neighbor_model_.kneighbors(X_validated)
        class_probabilities = np.zeros((X_validated.shape[0], len(self.classes_)))
        weight_power = 2.0 / (self.m - 1.0)

        for sample_index in range(X_validated.shape[0]):
            sample_distances = distances[sample_index]
            sample_neighbors = neighbor_indices[sample_index]
            neighbor_memberships = self.training_membership_[sample_neighbors]

            zero_distance_mask = sample_distances == 0.0
            if zero_distance_mask.any():
                exact_memberships = neighbor_memberships[zero_distance_mask]
                class_probabilities[sample_index] = exact_memberships.mean(axis=0)
                continue

            weights = 1.0 / np.power(sample_distances, weight_power)
            weighted_membership = neighbor_memberships * weights[:, np.newaxis]
            membership_sum = weighted_membership.sum(axis=0)
            normalization = membership_sum.sum()
            if normalization == 0.0:
                normalization = 1.0
            class_probabilities[sample_index] = membership_sum / normalization

        return class_probabilities

    def _build_training_membership(self, y: np.ndarray) -> np.ndarray:
        # Constrói pertinencias crisp one-hot para as amostras rotuladas do treino.
        membership = np.zeros((y.shape[0], len(self.classes_)), dtype=float)
        for sample_index, class_label in enumerate(y):
            class_index = self.class_to_index_[class_label]
            membership[sample_index, class_index] = 1.0
        return membership


def create_fuzzy_knn_classifier(
    n_neighbors: int = 5,
    m: float = 2.0,
    metric: str = "euclidean",
) -> FuzzyKNNClassifier:
    # Cria uma instancia configuravel do classificador Fuzzy KNN.
    return FuzzyKNNClassifier(
        n_neighbors=n_neighbors,
        m=m,
        metric=metric,
    )


def get_fuzzy_knn_param_grid() -> list[dict[str, Any]]:
    # Retorna uma grade pequena e viavel para os experimentos futuros.
    return [
        {"n_neighbors": 3, "m": 1.5},
        {"n_neighbors": 5, "m": 1.5},
        {"n_neighbors": 5, "m": 2.0},
        {"n_neighbors": 9, "m": 2.0},
        {"n_neighbors": 9, "m": 2.5},
    ]


def train_fuzzy_knn_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **model_params: Any,
) -> tuple[FuzzyKNNClassifier, float]:
    # Ajusta o classificador Fuzzy KNN e retorna o tempo gasto.
    started_at = perf_counter()
    model = create_fuzzy_knn_classifier(**model_params)
    model.fit(X_train, y_train)
    training_time_seconds = perf_counter() - started_at
    return model, training_time_seconds


def evaluate_fuzzy_knn_classifier(
    model: FuzzyKNNClassifier,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    # Avalia o Fuzzy KNN apenas no conjunto de validacao.
    y_pred = model.predict(X_validation)
    y_proba = model.predict_proba(X_validation)
    return {
        "validation_accuracy": float(accuracy_score(y_validation, y_pred)),
        "predictions_shape": y_pred.shape,
        "probabilities_shape": y_proba.shape,
    }


def run_fuzzy_knn_smoke_test(
    dataset_name: str,
    seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Executa um teste rapido do Fuzzy KNN usando treino e validacao.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    default_params = {
        "n_neighbors": 5,
        "m": 2.0,
        "metric": "euclidean",
    }
    if model_params:
        default_params.update(model_params)

    X_train = prepared["processed_splits"]["X_train"]
    y_train = prepared["raw_splits"]["y_train"]
    X_validation = prepared["processed_splits"]["X_validation"]
    y_validation = prepared["raw_splits"]["y_validation"]

    model, training_time_seconds = train_fuzzy_knn_classifier(
        X_train=X_train,
        y_train=y_train,
        **default_params,
    )
    evaluation = evaluate_fuzzy_knn_classifier(
        model=model,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    return {
        "dataset_name": prepared["dataset_name"],
        "display_name": prepared["display_name"],
        "seed": seed,
        "classes": model.classes_.tolist(),
        "model_params": default_params,
        "training_time_seconds": training_time_seconds,
        "train_shape": X_train.shape,
        "validation_shape": X_validation.shape,
        "test_shape": prepared["processed_splits"]["X_test"].shape,
        **evaluation,
    }


def format_fuzzy_knn_smoke_test_report(result: dict[str, Any]) -> str:
    # Formata um resumo textual curto do smoke test do Fuzzy KNN.
    lines = [
        "Teste do Fuzzy KNN concluido com sucesso.",
        f"Dataset: {result['display_name']} ({result['dataset_name']})",
        f"Seed: {result['seed']}",
        f"Classes: {result['classes']}",
        (
            f"Shapes -> treino: {result['train_shape']}, "
            f"validacao: {result['validation_shape']}, "
            f"teste preservado: {result['test_shape']}"
        ),
        f"Acuracia de validacao: {result['validation_accuracy']:.4f}",
        f"Tempo de treino: {result['training_time_seconds']:.4f} s",
        f"Shape de predict_proba: {result['probabilities_shape']}",
        f"Parametros: {result['model_params']}",
    ]
    return "\n".join(lines)
