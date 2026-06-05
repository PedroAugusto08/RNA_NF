from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.config import DEFAULT_RANDOM_SEED
from src.preprocessing import prepare_dataset_splits


class ZeroOrderSugenoClassifier(BaseEstimator, ClassifierMixin):
    # Classificador fuzzy do tipo Sugeno de ordem zero com regras extraidas por clustering.

    def __init__(
        self,
        n_rules: int = 4,
        sigma_scale: float = 1.0,
        random_state: int | None = None,
        max_iter: int = 300,
        n_init: int = 10,
    ) -> None:
        self.n_rules = n_rules
        self.sigma_scale = sigma_scale
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "ZeroOrderSugenoClassifier":
        # Ajusta regras fuzzy a partir de clusters e estima consequentes constantes por classe.
        X_validated, y_validated = check_X_y(X, y, dtype=float)

        if self.n_rules < 1:
            raise ValueError("n_rules deve ser maior ou igual a 1.")
        if self.n_rules > X_validated.shape[0]:
            raise ValueError(
                "n_rules nao pode ser maior que o numero de amostras de treino."
            )
        if self.sigma_scale <= 0.0:
            raise ValueError("sigma_scale deve ser maior que 0.")
        if self.n_init < 1:
            raise ValueError("n_init deve ser maior ou igual a 1.")

        self.classes_ = np.unique(y_validated)
        self.class_to_index_ = {
            class_label: class_index
            for class_index, class_label in enumerate(self.classes_)
        }
        self.n_features_in_ = X_validated.shape[1]

        self.kmeans_ = KMeans(
            n_clusters=self.n_rules,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        cluster_indices = self.kmeans_.fit_predict(X_validated)
        self.centers_ = self.kmeans_.cluster_centers_
        self.sigmas_ = self._estimate_rule_sigmas(X_validated, cluster_indices)
        self.rule_consequents_ = self._build_rule_consequents(y_validated, cluster_indices)
        self.rule_weights_ = self._build_rule_weights(cluster_indices)
        self.rules_ = self._build_rules()
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Prediz a classe com maior ativacao agregada na saida Sugeno.
        class_probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(class_probabilities, axis=1)
        return self.classes_[predicted_indices]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Calcula saidas normalizadas a partir das regras gaussianas e consequentes constantes.
        check_is_fitted(
            self,
            ["centers_", "sigmas_", "rule_consequents_", "rule_weights_", "classes_"],
        )
        X_validated = check_array(X, dtype=float)
        if X_validated.shape[1] != self.n_features_in_:
            raise ValueError(
                "Numero de atributos inconsistente com o conjunto de treino."
            )

        activations = self._compute_rule_activations(X_validated)
        weighted_outputs = activations[:, :, np.newaxis] * self.rule_consequents_[np.newaxis, :, :]
        class_scores = weighted_outputs.sum(axis=1)
        normalization = class_scores.sum(axis=1, keepdims=True)
        normalization[normalization == 0.0] = 1.0
        return class_scores / normalization

    def _estimate_rule_sigmas(
        self,
        X: np.ndarray,
        cluster_indices: np.ndarray,
    ) -> np.ndarray:
        # Estima um desvio por regra a partir das distancias intra-cluster.
        sigmas = np.zeros(self.n_rules, dtype=float)
        global_scale = np.mean(np.std(X, axis=0))
        if global_scale <= 0.0:
            global_scale = 1.0

        for rule_index in range(self.n_rules):
            cluster_mask = cluster_indices == rule_index
            cluster_samples = X[cluster_mask]
            center = self.centers_[rule_index]

            if cluster_samples.shape[0] <= 1:
                sigma_value = global_scale
            else:
                squared_distances = np.sum((cluster_samples - center) ** 2, axis=1)
                sigma_value = float(np.sqrt(np.mean(squared_distances)))
                if sigma_value <= 0.0:
                    sigma_value = global_scale

            sigmas[rule_index] = max(sigma_value * self.sigma_scale, 1e-6)

        return sigmas

    def _build_rule_consequents(
        self,
        y: np.ndarray,
        cluster_indices: np.ndarray,
    ) -> np.ndarray:
        # Estima consequentes constantes por classe como distribuicoes locais em cada regra.
        consequents = np.zeros((self.n_rules, len(self.classes_)), dtype=float)

        for rule_index in range(self.n_rules):
            cluster_mask = cluster_indices == rule_index
            if not np.any(cluster_mask):
                consequents[rule_index] = 1.0 / len(self.classes_)
                continue

            cluster_targets = y[cluster_mask]
            for class_index, class_label in enumerate(self.classes_):
                consequents[rule_index, class_index] = float(
                    np.sum(cluster_targets == class_label)
                )

            total = consequents[rule_index].sum()
            if total == 0.0:
                consequents[rule_index] = 1.0 / len(self.classes_)
            else:
                consequents[rule_index] /= total

        return consequents

    def _build_rule_weights(self, cluster_indices: np.ndarray) -> np.ndarray:
        # Usa o tamanho relativo do cluster como peso inicial das regras.
        counts = np.bincount(cluster_indices, minlength=self.n_rules).astype(float)
        total = counts.sum()
        if total == 0.0:
            return np.full(self.n_rules, 1.0 / self.n_rules)
        return counts / total

    def _compute_rule_activations(self, X: np.ndarray) -> np.ndarray:
        # Calcula ativacoes gaussianas das regras fuzzy para cada amostra.
        squared_distances = np.sum(
            (X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        sigma_squared = np.square(self.sigmas_)[np.newaxis, :]
        gaussian_membership = np.exp(-squared_distances / (2.0 * sigma_squared))
        weighted_membership = gaussian_membership * self.rule_weights_[np.newaxis, :]
        normalization = weighted_membership.sum(axis=1, keepdims=True)
        normalization[normalization == 0.0] = 1.0
        return weighted_membership / normalization

    def _build_rules(self) -> list[dict[str, Any]]:
        # Gera uma representacao interpretavel das regras fuzzy aprendidas.
        rules = []
        for rule_index in range(self.n_rules):
            consequent_distribution = {
                str(class_label): float(self.rule_consequents_[rule_index, class_index])
                for class_index, class_label in enumerate(self.classes_)
            }
            dominant_class = self.classes_[int(np.argmax(self.rule_consequents_[rule_index]))]
            rules.append(
                {
                    "rule_index": rule_index,
                    "dominant_class": dominant_class,
                    "center": self.centers_[rule_index].tolist(),
                    "sigma": float(self.sigmas_[rule_index]),
                    "weight": float(self.rule_weights_[rule_index]),
                    "consequent_distribution": consequent_distribution,
                }
            )
        return rules


def create_zero_order_sugeno_classifier(
    n_rules: int = 4,
    sigma_scale: float = 1.0,
    random_state: int | None = None,
    max_iter: int = 300,
    n_init: int = 10,
) -> ZeroOrderSugenoClassifier:
    # Cria uma instancia configuravel do classificador Sugeno de ordem zero.
    return ZeroOrderSugenoClassifier(
        n_rules=n_rules,
        sigma_scale=sigma_scale,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )


def get_zero_order_sugeno_param_grid(n_classes: int) -> list[dict[str, Any]]:
    # Retorna uma grade pequena para numero de regras e largura das gaussianas.
    if n_classes < 1:
        raise ValueError("n_classes deve ser maior ou igual a 1.")

    candidate_rules = []
    for value in (n_classes, 2 * n_classes, 4 * n_classes):
        if value not in candidate_rules:
            candidate_rules.append(value)

    return [
        {"n_rules": candidate_rules[0], "sigma_scale": 0.5},
        {"n_rules": candidate_rules[1], "sigma_scale": 0.5},
        {"n_rules": candidate_rules[1], "sigma_scale": 1.0},
        {"n_rules": candidate_rules[-1], "sigma_scale": 1.0},
        {"n_rules": candidate_rules[1], "sigma_scale": 2.0},
    ]


def train_zero_order_sugeno_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **model_params: Any,
) -> tuple[ZeroOrderSugenoClassifier, float]:
    # Treina o classificador Sugeno de ordem zero e retorna o tempo gasto.
    started_at = perf_counter()
    model = create_zero_order_sugeno_classifier(**model_params)
    model.fit(X_train, y_train)
    training_time_seconds = perf_counter() - started_at
    return model, training_time_seconds


def evaluate_zero_order_sugeno_classifier(
    model: ZeroOrderSugenoClassifier,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    # Avalia o Sugeno de ordem zero apenas no conjunto de validacao.
    y_pred = model.predict(X_validation)
    y_proba = model.predict_proba(X_validation)
    return {
        "validation_accuracy": float(accuracy_score(y_validation, y_pred)),
        "predictions_shape": y_pred.shape,
        "probabilities_shape": y_proba.shape,
    }


def format_zero_order_sugeno_rules(
    model: ZeroOrderSugenoClassifier,
    feature_names: list[str],
    top_features: int = 3,
) -> list[str]:
    # Resume cada regra pelas features mais influentes do centro antecedente.
    rules = []
    for rule in model.rules_:
        center = np.asarray(rule["center"], dtype=float)
        top_indices = np.argsort(np.abs(center))[::-1][:top_features]
        feature_summary = ", ".join(
            f"{feature_names[index]}={center[index]:.4f}" for index in top_indices
        )
        rules.append(
            f"regra {rule['rule_index']} -> classe {rule['dominant_class']} | "
            f"sigma={rule['sigma']:.4f}, peso={rule['weight']:.4f} | "
            f"{feature_summary} | consequente {rule['consequent_distribution']}"
        )
    return rules


def run_zero_order_sugeno_smoke_test(
    dataset_name: str,
    seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Executa um teste rapido do Sugeno de ordem zero usando treino e validacao.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    y_train = prepared["raw_splits"]["y_train"]
    default_params = {
        "n_rules": max(len(y_train.unique()) * 2, 2),
        "sigma_scale": 1.0,
        "random_state": seed,
        "max_iter": 300,
        "n_init": 10,
    }
    if model_params:
        default_params.update(model_params)

    X_train = prepared["processed_splits"]["X_train"]
    X_validation = prepared["processed_splits"]["X_validation"]
    y_validation = prepared["raw_splits"]["y_validation"]

    model, training_time_seconds = train_zero_order_sugeno_classifier(
        X_train=X_train,
        y_train=y_train,
        **default_params,
    )
    evaluation = evaluate_zero_order_sugeno_classifier(
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
        "rules": format_zero_order_sugeno_rules(model, prepared["feature_names"]),
        **evaluation,
    }


def format_zero_order_sugeno_smoke_test_report(result: dict[str, Any]) -> str:
    # Formata um resumo textual curto do smoke test do Sugeno de ordem zero.
    lines = [
        "Teste do Sugeno de ordem zero concluido com sucesso.",
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
        "Regras geradas:",
    ]
    for rule in result["rules"]:
        lines.append(f"  - {rule}")
    return "\n".join(lines)
