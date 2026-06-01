from __future__ import annotations

from time import perf_counter
from typing import Any
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from src.config import DEFAULT_RANDOM_SEED
from src.preprocessing import prepare_dataset_splits


DEFAULT_MLP_PARAMS = {
    "hidden_layer_sizes": (64,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 500,
}


def create_mlp_classifier(
    hidden_layer_sizes: tuple[int, ...] = (64,),
    activation: str = "relu",
    solver: str = "adam",
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    max_iter: int = 500,
    random_state: int | None = None,
) -> MLPClassifier:
    # Cria uma instancia configuravel do MLPClassifier.
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
    )


def get_mlp_param_grid() -> list[dict[str, Any]]:
    # Retorna uma grade pequena e justificavel para os experimentos futuros.
    return [
        {
            "hidden_layer_sizes": (32,),
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
        },
        {
            "hidden_layer_sizes": (64,),
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
        },
        {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
        },
        {
            "hidden_layer_sizes": (64,),
            "activation": "tanh",
            "alpha": 0.0001,
            "learning_rate_init": 0.01,
            "max_iter": 500,
        },
        {
            "hidden_layer_sizes": (64, 32),
            "activation": "tanh",
            "alpha": 0.001,
            "learning_rate_init": 0.01,
            "max_iter": 500,
        },
    ]


def train_mlp_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **model_params: Any,
) -> tuple[MLPClassifier, float, list[str]]:
    # Treina o MLP e retorna o modelo com o tempo gasto.
    started_at = perf_counter()
    model = create_mlp_classifier(**model_params)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    training_time_seconds = perf_counter() - started_at

    warning_messages = []
    for captured_warning in captured_warnings:
        if issubclass(captured_warning.category, ConvergenceWarning):
            warning_messages.append(str(captured_warning.message))

    return model, training_time_seconds, warning_messages


def evaluate_mlp_classifier(
    model: MLPClassifier,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    # Avalia o MLP apenas no conjunto de validacao.
    y_pred = model.predict(X_validation)
    result = {
        "validation_accuracy": float(accuracy_score(y_validation, y_pred)),
        "predictions_shape": y_pred.shape,
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_validation)
        result["probabilities_shape"] = y_proba.shape

    return result


def run_mlp_smoke_test(
    dataset_name: str,
    seed: int = DEFAULT_RANDOM_SEED,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Executa um teste rapido do MLP usando apenas treino e validacao.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    params = dict(DEFAULT_MLP_PARAMS)
    if model_params:
        params.update(model_params)
    params["random_state"] = seed

    X_train = prepared["processed_splits"]["X_train"]
    y_train = prepared["raw_splits"]["y_train"]
    X_validation = prepared["processed_splits"]["X_validation"]
    y_validation = prepared["raw_splits"]["y_validation"]

    model, training_time_seconds, warning_messages = train_mlp_classifier(
        X_train=X_train,
        y_train=y_train,
        **params,
    )
    evaluation = evaluate_mlp_classifier(
        model=model,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    return {
        "dataset_name": prepared["dataset_name"],
        "display_name": prepared["display_name"],
        "seed": seed,
        "classes": model.classes_.tolist(),
        "model_params": params,
        "training_time_seconds": training_time_seconds,
        "warnings": warning_messages,
        "train_shape": X_train.shape,
        "validation_shape": X_validation.shape,
        "test_shape": prepared["processed_splits"]["X_test"].shape,
        **evaluation,
    }


def format_mlp_smoke_test_report(result: dict[str, Any]) -> str:
    # Formata um resumo textual curto do smoke test do MLP.
    lines = [
        "Teste do MLP concluido com sucesso.",
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
        f"Parametros: {result['model_params']}",
    ]

    if "probabilities_shape" in result:
        lines.append(f"Shape de predict_proba: {result['probabilities_shape']}")

    if result["warnings"]:
        lines.append(f"Avisos capturados: {result['warnings']}")

    return "\n".join(lines)
