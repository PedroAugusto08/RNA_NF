from __future__ import annotations

from time import perf_counter
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RBFNetworkClassifier(BaseEstimator, ClassifierMixin):
    # Classificador RBF simplificado com centros obtidos por KMeans.

    def __init__(
        self,
        n_centers: int = 10,
        gamma: float | None = None,
        random_state: int | None = None,
        max_iter: int = 300,
        output_max_iter: int = 1000,
        kmeans_n_init: int = 10,
    ) -> None:
        self.n_centers = n_centers
        self.gamma = gamma
        self.random_state = random_state
        self.max_iter = max_iter
        self.output_max_iter = output_max_iter
        self.kmeans_n_init = kmeans_n_init

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "RBFNetworkClassifier":
        # Ajusta o KMeans, calcula ativacoes RBF e treina a camada de saida.
        X_validated, y_validated = check_X_y(X, y, dtype=float)

        if self.n_centers < 1:
            raise ValueError("n_centers deve ser maior ou igual a 1.")

        if self.n_centers > X_validated.shape[0]:
            raise ValueError(
                "n_centers nao pode ser maior que o numero de amostras de treino."
            )

        self.kmeans_ = KMeans(
            n_clusters=self.n_centers,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.kmeans_n_init,
        )
        self.kmeans_.fit(X_validated)

        self.centers_ = self.kmeans_.cluster_centers_
        self.gamma_ = self._resolve_gamma(X_validated)
        self.classes_ = np.unique(y_validated)

        X_rbf = self._rbf_transform(X_validated)
        self.output_model_ = LogisticRegression(
            max_iter=self.output_max_iter,
            random_state=self.random_state,
        )
        self.output_model_.fit(X_rbf, y_validated)
        self.n_features_in_ = X_validated.shape[1]
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Prediz as classes a partir das ativacoes RBF.
        check_is_fitted(self, ["centers_", "gamma_", "output_model_"])
        X_validated = check_array(X, dtype=float)
        X_rbf = self._rbf_transform(X_validated)
        return self.output_model_.predict(X_rbf)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Prediz probabilidades usando a camada de saida linear.
        check_is_fitted(self, ["centers_", "gamma_", "output_model_"])
        X_validated = check_array(X, dtype=float)
        X_rbf = self._rbf_transform(X_validated)
        return self.output_model_.predict_proba(X_rbf)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        # Resolve gamma automaticamente a partir das distancias treino-centros.
        if self.gamma is not None:
            if self.gamma <= 0:
                raise ValueError("gamma deve ser positivo quando informado.")
            return float(self.gamma)

        squared_distances = self._squared_distances(X, self.centers_)
        min_squared_distances = squared_distances.min(axis=1)
        sigma_squared = float(np.mean(min_squared_distances))
        sigma_squared = max(sigma_squared, 1e-12)
        return 1.0 / (2.0 * sigma_squared)

    def _rbf_transform(self, X: np.ndarray) -> np.ndarray:
        # Converte as amostras em ativacoes gaussianas em relacao aos centros.
        squared_distances = self._squared_distances(X, self.centers_)
        return np.exp(-self.gamma_ * squared_distances)

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # Calcula distancias euclidianas ao quadrado entre amostras e centros.
        difference = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return np.sum(difference * difference, axis=2)


def create_rbf_network_classifier(
    n_centers: int = 10,
    gamma: float | None = None,
    random_state: int | None = None,
    max_iter: int = 300,
    output_max_iter: int = 1000,
    kmeans_n_init: int = 10,
) -> RBFNetworkClassifier:
    # Cria uma instancia configuravel da RBF simplificada.
    return RBFNetworkClassifier(
        n_centers=n_centers,
        gamma=gamma,
        random_state=random_state,
        max_iter=max_iter,
        output_max_iter=output_max_iter,
        kmeans_n_init=kmeans_n_init,
    )


def get_rbf_param_grid(n_classes: int) -> list[dict[str, Any]]:
    # Gera uma grade pequena baseada no numero de classes do dataset.
    if n_classes < 1:
        raise ValueError("n_classes deve ser maior ou igual a 1.")

    candidate_centers = []
    for value in (n_classes, 2 * n_classes, 4 * n_classes):
        if value not in candidate_centers:
            candidate_centers.append(value)

    return [
        {"n_centers": candidate_centers[0], "gamma": None},
        {"n_centers": candidate_centers[1], "gamma": None},
        {"n_centers": candidate_centers[-1], "gamma": None},
        {"n_centers": candidate_centers[1], "gamma": 0.1},
        {"n_centers": candidate_centers[1], "gamma": 1.0},
    ]


def train_rbf_network_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **model_params: Any,
) -> tuple[RBFNetworkClassifier, float, list[str]]:
    # Treina a RBF simplificada e retorna o modelo com o tempo gasto.
    started_at = perf_counter()
    model = create_rbf_network_classifier(**model_params)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    training_time_seconds = perf_counter() - started_at

    warning_messages = []
    for captured_warning in captured_warnings:
        if issubclass(captured_warning.category, ConvergenceWarning):
            warning_messages.append(str(captured_warning.message))

    return model, training_time_seconds, warning_messages


def evaluate_rbf_network_classifier(
    model: RBFNetworkClassifier,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    # Avalia a RBF simplificada apenas no conjunto de validacao.
    y_pred = model.predict(X_validation)
    y_proba = model.predict_proba(X_validation)
    return {
        "validation_accuracy": float(accuracy_score(y_validation, y_pred)),
        "predictions_shape": y_pred.shape,
        "probabilities_shape": y_proba.shape,
    }
