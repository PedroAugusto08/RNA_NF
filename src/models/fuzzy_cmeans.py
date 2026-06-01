from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import skfuzzy as fuzz

from src.config import DEFAULT_RANDOM_SEED
from src.preprocessing import prepare_dataset_splits


class FuzzyCMeansClassifier(BaseEstimator, ClassifierMixin):
    # Classificador fuzzy baseado em clusters do Fuzzy C-Means.

    def __init__(
        self,
        n_clusters: int = 4,
        m: float = 2.0,
        error: float = 1e-5,
        max_iter: int = 300,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "FuzzyCMeansClassifier":
        # Ajusta os clusters fuzzy e associa cada cluster a uma classe predominante.
        X_validated, y_validated = check_X_y(X, y, dtype=float)

        if self.n_clusters < 1:
            raise ValueError("n_clusters deve ser maior ou igual a 1.")
        if self.n_clusters > X_validated.shape[0]:
            raise ValueError(
                "n_clusters nao pode ser maior que o numero de amostras de treino."
            )
        if self.m <= 1.0:
            raise ValueError("m deve ser maior que 1.0 no Fuzzy C-Means.")

        self.classes_ = np.unique(y_validated)
        self.class_to_index_ = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.feature_count_ = X_validated.shape[1]
        initial_membership = self._build_initial_membership(X_validated)

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=X_validated.T,
            c=self.n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            init=initial_membership,
        )

        self.centers_ = cntr
        self.membership_train_ = u
        self.initial_membership_ = u0
        self.distance_train_ = d
        self.objective_function_history_ = jm
        self.iterations_ = p
        self.fpc_ = float(fpc)

        self.cluster_labels_ = self._assign_cluster_labels(y_validated, u)
        self.cluster_class_distribution_ = self._build_cluster_class_distribution(
            y_validated,
            u,
        )
        self.rules_ = self._build_rules()
        return self

    def _build_initial_membership(self, X: np.ndarray) -> np.ndarray:
        # Inicializa as pertinencias a partir de clusters hard do KMeans.
        if self.n_clusters == 1:
            return np.ones((1, X.shape[0]), dtype=float)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=self.max_iter,
        )
        labels = kmeans.fit_predict(X)

        low_weight = 0.1 / (self.n_clusters - 1)
        initial_membership = np.full(
            (self.n_clusters, X.shape[0]),
            low_weight,
            dtype=float,
        )
        for sample_index, cluster_index in enumerate(labels):
            initial_membership[cluster_index, sample_index] = 0.9

        column_sums = initial_membership.sum(axis=0, keepdims=True)
        return initial_membership / column_sums

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Prediz classes pela maior pertinencia agregada por classe.
        class_probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(class_probabilities, axis=1)
        return self.classes_[predicted_indices]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Prediz probabilidades agregando as pertinencias dos clusters por classe.
        check_is_fitted(
            self,
            [
                "centers_",
                "classes_",
                "cluster_labels_",
                "cluster_class_distribution_",
            ],
        )
        X_validated = check_array(X, dtype=float)
        if X_validated.shape[1] != self.feature_count_:
            raise ValueError(
                "Numero de atributos inconsistente com o conjunto de treino."
            )

        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            test_data=X_validated.T,
            cntr_trained=self.centers_,
            m=self.m,
            error=self.error,
            maxiter=self.max_iter,
            seed=self.random_state,
        )

        class_probabilities = np.zeros((X_validated.shape[0], len(self.classes_)))
        for cluster_index in range(self.n_clusters):
            class_weights = self.cluster_class_distribution_[cluster_index]
            class_probabilities += np.outer(u[cluster_index], class_weights)

        row_sums = class_probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return class_probabilities / row_sums

    def _assign_cluster_labels(self, y: np.ndarray, membership: np.ndarray) -> np.ndarray:
        # Associa cada cluster a classe com maior massa total de pertinencia.
        cluster_labels = []
        for cluster_index in range(self.n_clusters):
            membership_scores = {}
            for class_label in self.classes_:
                class_mask = y == class_label
                membership_scores[class_label] = float(
                    membership[cluster_index, class_mask].sum()
                )
            best_label = max(
                membership_scores,
                key=lambda class_label: membership_scores[class_label],
            )
            cluster_labels.append(best_label)
        return np.array(cluster_labels, dtype=object)

    def _build_cluster_class_distribution(
        self,
        y: np.ndarray,
        membership: np.ndarray,
    ) -> np.ndarray:
        # Constrói pesos normalizados de classes para cada cluster.
        distributions = np.zeros((self.n_clusters, len(self.classes_)))
        for cluster_index in range(self.n_clusters):
            for class_index, class_label in enumerate(self.classes_):
                class_mask = y == class_label
                distributions[cluster_index, class_index] = float(
                    membership[cluster_index, class_mask].sum()
                )

        row_sums = distributions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return distributions / row_sums

    def _build_rules(self) -> list[dict[str, Any]]:
        # Gera uma representacao interpretavel das regras cluster -> classe.
        rules = []
        for cluster_index in range(self.n_clusters):
            class_distribution = {
                str(class_label): float(
                    self.cluster_class_distribution_[cluster_index, class_index]
                )
                for class_index, class_label in enumerate(self.classes_)
            }
            rules.append(
                {
                    "cluster_index": cluster_index,
                    "assigned_class": self.cluster_labels_[cluster_index],
                    "center": self.centers_[cluster_index].tolist(),
                    "class_distribution": class_distribution,
                }
            )
        return rules


def create_fuzzy_cmeans_classifier(
    n_clusters: int = 4,
    m: float = 2.0,
    error: float = 1e-5,
    max_iter: int = 300,
    random_state: int | None = None,
) -> FuzzyCMeansClassifier:
    # Cria uma instancia configuravel do classificador Fuzzy C-Means.
    return FuzzyCMeansClassifier(
        n_clusters=n_clusters,
        m=m,
        error=error,
        max_iter=max_iter,
        random_state=random_state,
    )


def get_fuzzy_cmeans_param_grid(n_classes: int) -> list[dict[str, Any]]:
    # Retorna uma grade pequena para variacao de clusters e parametro m.
    if n_classes < 1:
        raise ValueError("n_classes deve ser maior ou igual a 1.")

    candidate_clusters = []
    for value in (n_classes, 2 * n_classes, 4 * n_classes):
        if value not in candidate_clusters:
            candidate_clusters.append(value)

    return [
        {"n_clusters": candidate_clusters[0], "m": 1.5},
        {"n_clusters": candidate_clusters[1], "m": 1.5},
        {"n_clusters": candidate_clusters[1], "m": 2.0},
        {"n_clusters": candidate_clusters[-1], "m": 2.0},
        {"n_clusters": candidate_clusters[1], "m": 2.5},
    ]


def train_fuzzy_cmeans_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **model_params: Any,
) -> tuple[FuzzyCMeansClassifier, float]:
    # Treina o classificador Fuzzy C-Means e retorna o tempo gasto.
    started_at = perf_counter()
    model = create_fuzzy_cmeans_classifier(**model_params)
    model.fit(X_train, y_train)
    training_time_seconds = perf_counter() - started_at
    return model, training_time_seconds


def evaluate_fuzzy_cmeans_classifier(
    model: FuzzyCMeansClassifier,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    # Avalia o Fuzzy C-Means apenas no conjunto de validacao.
    y_pred = model.predict(X_validation)
    y_proba = model.predict_proba(X_validation)
    return {
        "validation_accuracy": float(accuracy_score(y_validation, y_pred)),
        "predictions_shape": y_pred.shape,
        "probabilities_shape": y_proba.shape,
    }


def format_fuzzy_cmeans_rules(
    model: FuzzyCMeansClassifier,
    feature_names: list[str],
    top_features: int = 3,
) -> list[str]:
    # Resume cada cluster pelas features de maior magnitude no centro.
    rules = []
    for rule in model.rules_:
        center = np.array(rule["center"], dtype=float)
        top_indices = np.argsort(np.abs(center))[::-1][:top_features]
        feature_summary = ", ".join(
            f"{feature_names[index]}={center[index]:.4f}" for index in top_indices
        )
        rules.append(
            f"cluster {rule['cluster_index']} -> classe {rule['assigned_class']} | "
            f"{feature_summary} | distribuicao {rule['class_distribution']}"
        )
    return rules


def run_fuzzy_cmeans_smoke_test(
    dataset_name: str,
    seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Executa um teste rapido do Fuzzy C-Means usando treino e validacao.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    y_train = prepared["raw_splits"]["y_train"]
    default_params = {
        "n_clusters": max(len(y_train.unique()) * 2, 2),
        "m": 2.0,
        "error": 1e-5,
        "max_iter": 300,
        "random_state": seed,
    }
    if model_params:
        default_params.update(model_params)

    X_train = prepared["processed_splits"]["X_train"]
    X_validation = prepared["processed_splits"]["X_validation"]
    y_validation = prepared["raw_splits"]["y_validation"]

    model, training_time_seconds = train_fuzzy_cmeans_classifier(
        X_train=X_train,
        y_train=y_train,
        **default_params,
    )
    evaluation = evaluate_fuzzy_cmeans_classifier(
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
        "rules": format_fuzzy_cmeans_rules(model, prepared["feature_names"]),
        "fpc": model.fpc_,
        **evaluation,
    }


def format_fuzzy_cmeans_smoke_test_report(result: dict[str, Any]) -> str:
    # Formata um resumo textual curto do smoke test do Fuzzy C-Means.
    lines = [
        "Teste do Fuzzy C-Means concluido com sucesso.",
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
        f"FPC: {result['fpc']:.6f}",
        f"Parametros: {result['model_params']}",
        "Regras geradas:",
    ]
    for rule in result["rules"]:
        lines.append(f"  - {rule}")
    return "\n".join(lines)
