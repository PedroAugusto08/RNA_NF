from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATASET_CONFIGS,
    DEFAULT_EXPERIMENT_RUNS,
    DEFAULT_RANDOM_SEED,
    EXPERIMENTS_DIR,
)
from src.models import (
    evaluate_fuzzy_cmeans_classifier,
    evaluate_fuzzy_knn_classifier,
    evaluate_mlp_classifier,
    evaluate_rbf_network_classifier,
    train_fuzzy_cmeans_classifier,
    train_fuzzy_knn_classifier,
    train_mlp_classifier,
    train_rbf_network_classifier,
)
from src.preprocessing import prepare_dataset_splits


ALGORITHM_NAMES = [
    "mlp_classifier",
    "rbf_network",
    "fuzzy_cmeans",
    "fuzzy_knn",
]


def ensure_experiments_directory() -> None:
    # Garante que a pasta de artefatos experimentais exista.
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_seeds(
    n_runs: int = DEFAULT_EXPERIMENT_RUNS,
    start_seed: int = DEFAULT_RANDOM_SEED,
) -> list[int]:
    # Gera seeds consecutivas para as execucoes independentes.
    if n_runs < 1:
        raise ValueError("n_runs deve ser maior ou igual a 1.")
    return [start_seed + offset for offset in range(n_runs)]


def serialize_model_params(model_params: dict[str, Any]) -> str:
    # Serializa hiperparametros para salvar em tabela CSV.
    return json.dumps(model_params, sort_keys=True, default=str)


def build_split_signature(split_indices: dict[str, pd.Index]) -> str:
    # Cria uma assinatura compacta do particionamento usado naquela seed.
    signature_payload = "|".join(
        [
            ",".join(map(str, split_indices["train"].tolist())),
            ",".join(map(str, split_indices["validation"].tolist())),
            ",".join(map(str, split_indices["test"].tolist())),
        ]
    )
    return hashlib.md5(signature_payload.encode("utf-8")).hexdigest()


def get_default_algorithm_params(
    algorithm_name: str,
    n_classes: int,
    seed: int,
) -> dict[str, Any]:
    # Define uma configuracao padrao inicial para cada algoritmo.
    if algorithm_name == "mlp_classifier":
        return {
            "hidden_layer_sizes": (64,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "random_state": seed,
        }

    if algorithm_name == "rbf_network":
        return {
            "n_centers": max(2 * n_classes, 2),
            "gamma": None,
            "random_state": seed,
            "max_iter": 300,
            "output_max_iter": 1000,
        }

    if algorithm_name == "fuzzy_cmeans":
        return {
            "n_clusters": max(2 * n_classes, 2),
            "m": 2.0,
            "error": 1e-5,
            "max_iter": 300,
            "random_state": seed,
        }

    if algorithm_name == "fuzzy_knn":
        return {
            "n_neighbors": 5,
            "m": 2.0,
            "metric": "euclidean",
        }

    raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")


def run_single_algorithm(
    algorithm_name: str,
    prepared: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    # Treina e avalia um algoritmo em um split ja preparado.
    X_train = prepared["processed_splits"]["X_train"]
    y_train = prepared["raw_splits"]["y_train"]
    X_validation = prepared["processed_splits"]["X_validation"]
    y_validation = prepared["raw_splits"]["y_validation"]
    n_classes = int(y_train.nunique())
    model_params = get_default_algorithm_params(
        algorithm_name=algorithm_name,
        n_classes=n_classes,
        seed=seed,
    )

    warnings_list: list[str] = []

    if algorithm_name == "mlp_classifier":
        model, training_time_seconds, warnings_list = train_mlp_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        evaluation = evaluate_mlp_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )
    elif algorithm_name == "rbf_network":
        model, training_time_seconds, warnings_list = train_rbf_network_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        evaluation = evaluate_rbf_network_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )
    elif algorithm_name == "fuzzy_cmeans":
        model, training_time_seconds = train_fuzzy_cmeans_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        evaluation = evaluate_fuzzy_cmeans_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )
    elif algorithm_name == "fuzzy_knn":
        model, training_time_seconds = train_fuzzy_knn_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        evaluation = evaluate_fuzzy_knn_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")

    split_signature = build_split_signature(prepared["split_indices"])
    return {
        "dataset_name": prepared["dataset_name"],
        "display_name": prepared["display_name"],
        "algorithm_name": algorithm_name,
        "seed": seed,
        "split_signature": split_signature,
        "n_classes": n_classes,
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_validation.shape[0]),
        "test_samples": int(prepared["processed_splits"]["X_test"].shape[0]),
        "train_features": int(X_train.shape[1]),
        "validation_accuracy": float(evaluation["validation_accuracy"]),
        "training_time_seconds": float(training_time_seconds),
        "warning_count": len(warnings_list),
        "warnings": " | ".join(warnings_list),
        "model_params": serialize_model_params(model_params),
    }


def run_dataset_experiments_for_seed(
    dataset_name: str,
    seed: int,
    algorithm_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    # Executa todos os algoritmos sobre o mesmo split de uma seed.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    selected_algorithms = algorithm_names or list(ALGORITHM_NAMES)
    return [
        run_single_algorithm(
            algorithm_name=algorithm_name,
            prepared=prepared,
            seed=seed,
        )
        for algorithm_name in selected_algorithms
    ]


def build_runs_dataframe(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    # Converte as execucoes individuais em DataFrame tabular.
    return pd.DataFrame(run_records)


def build_summary_dataframe(runs_df: pd.DataFrame) -> pd.DataFrame:
    # Consolida media e desvio-padrao por dataset e algoritmo.
    summary_df = (
        runs_df.groupby(["dataset_name", "display_name", "algorithm_name"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_accuracy_std=("validation_accuracy", "std"),
            training_time_mean_seconds=("training_time_seconds", "mean"),
            training_time_std_seconds=("training_time_seconds", "std"),
            train_samples=("train_samples", "first"),
            validation_samples=("validation_samples", "first"),
            test_samples=("test_samples", "first"),
            train_features=("train_features", "first"),
        )
    )
    summary_df["validation_accuracy_std"] = summary_df[
        "validation_accuracy_std"
    ].fillna(0.0)
    summary_df["training_time_std_seconds"] = summary_df[
        "training_time_std_seconds"
    ].fillna(0.0)
    return summary_df.sort_values(
        by=["dataset_name", "validation_accuracy_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)


def save_experiment_results(
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, str]:
    # Salva as tabelas principais da ETAPA 8.
    ensure_experiments_directory()
    runs_path = EXPERIMENTS_DIR / "validation_runs.csv"
    summary_path = EXPERIMENTS_DIR / "validation_summary.csv"
    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return {
        "runs_path": str(runs_path),
        "summary_path": str(summary_path),
    }


def run_experiments(
    dataset_names: list[str] | None = None,
    n_runs: int = DEFAULT_EXPERIMENT_RUNS,
    start_seed: int = DEFAULT_RANDOM_SEED,
    algorithm_names: list[str] | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    # Executa multiplas seeds para um ou mais datasets e consolida os resultados.
    selected_datasets = dataset_names or list(DATASET_CONFIGS)
    seeds = generate_seeds(n_runs=n_runs, start_seed=start_seed)

    run_records: list[dict[str, Any]] = []
    for dataset_name in selected_datasets:
        for seed in seeds:
            run_records.extend(
                run_dataset_experiments_for_seed(
                    dataset_name=dataset_name,
                    seed=seed,
                    algorithm_names=algorithm_names,
                )
            )

    runs_df = build_runs_dataframe(run_records)
    summary_df = build_summary_dataframe(runs_df)

    artifact_paths = {}
    if save_results:
        artifact_paths = save_experiment_results(runs_df=runs_df, summary_df=summary_df)

    return {
        "dataset_names": selected_datasets,
        "seeds": seeds,
        "algorithm_names": algorithm_names or list(ALGORITHM_NAMES),
        "runs_df": runs_df,
        "summary_df": summary_df,
        "artifact_paths": artifact_paths,
    }


def format_experiments_report(experiment_result: dict[str, Any]) -> str:
    # Formata um resumo textual curto da ETAPA 8.
    summary_df = experiment_result["summary_df"]
    lines = [
        "Execucao experimental concluida com sucesso.",
        f"Datasets: {experiment_result['dataset_names']}",
        f"Seeds: {experiment_result['seeds']}",
        f"Algoritmos: {experiment_result['algorithm_names']}",
    ]

    if experiment_result["artifact_paths"]:
        lines.append(
            f"Execucoes salvas em: {experiment_result['artifact_paths']['runs_path']}"
        )
        lines.append(
            f"Resumo salvo em: {experiment_result['artifact_paths']['summary_path']}"
        )

    lines.append("")
    lines.append("Resumo por dataset e algoritmo:")

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.display_name} | {row.algorithm_name}: "
            f"acc_val = {row.validation_accuracy_mean:.4f} +- "
            f"{row.validation_accuracy_std:.4f}, "
            f"tempo = {row.training_time_mean_seconds:.4f} +- "
            f"{row.training_time_std_seconds:.4f} s"
        )

    return "\n".join(lines)
