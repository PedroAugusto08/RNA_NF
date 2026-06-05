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
    get_algorithm_display_name,
)
from src.models import (
    evaluate_fuzzy_cmeans_classifier,
    evaluate_mlp_classifier,
    evaluate_rbf_network_classifier,
    evaluate_zero_order_sugeno_classifier,
    get_fuzzy_cmeans_param_grid,
    get_mlp_param_grid,
    get_rbf_param_grid,
    get_zero_order_sugeno_param_grid,
    train_fuzzy_cmeans_classifier,
    train_mlp_classifier,
    train_rbf_network_classifier,
    train_zero_order_sugeno_classifier,
)
from src.preprocessing import prepare_dataset_splits


ALGORITHM_NAMES = [
    "mlp_classifier",
    "rbf_network",
    "fuzzy_cmeans",
    "sugeno_o0",
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


def deserialize_model_params(serialized_model_params: str) -> dict[str, Any]:
    # Desserializa hiperparametros salvos em formato JSON.
    return json.loads(serialized_model_params)


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

    if algorithm_name == "sugeno_o0":
        return {
            "n_rules": max(2 * n_classes, 2),
            "sigma_scale": 1.0,
            "random_state": seed,
            "max_iter": 300,
            "n_init": 10,
        }

    raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")


def get_algorithm_param_grid(
    algorithm_name: str,
    n_classes: int,
    seed: int,
) -> list[dict[str, Any]]:
    # Retorna a grade de hiperparametros com seeds acopladas quando necessario.
    if algorithm_name == "mlp_classifier":
        base_grid = get_mlp_param_grid()
    elif algorithm_name == "rbf_network":
        base_grid = get_rbf_param_grid(n_classes)
    elif algorithm_name == "fuzzy_cmeans":
        base_grid = get_fuzzy_cmeans_param_grid(n_classes)
    elif algorithm_name == "sugeno_o0":
        base_grid = get_zero_order_sugeno_param_grid(n_classes)
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")

    resolved_grid = []
    for params in base_grid:
        resolved_params = dict(params)
        if algorithm_name in {
            "mlp_classifier",
            "rbf_network",
            "fuzzy_cmeans",
            "sugeno_o0",
        }:
            resolved_params.setdefault("random_state", seed)
        resolved_grid.append(resolved_params)
    return resolved_grid


def fit_algorithm_model(
    algorithm_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict[str, Any],
) -> tuple[Any, float, list[str]]:
    # Treina um algoritmo e retorna o modelo ajustado, tempo e avisos.
    warnings_list: list[str] = []

    if algorithm_name == "mlp_classifier":
        model, training_time_seconds, warnings_list = train_mlp_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        return model, training_time_seconds, warnings_list

    if algorithm_name == "rbf_network":
        model, training_time_seconds, warnings_list = train_rbf_network_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        return model, training_time_seconds, warnings_list

    if algorithm_name == "fuzzy_cmeans":
        model, training_time_seconds = train_fuzzy_cmeans_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        return model, training_time_seconds, warnings_list

    if algorithm_name == "sugeno_o0":
        model, training_time_seconds = train_zero_order_sugeno_classifier(
            X_train=X_train,
            y_train=y_train,
            **model_params,
        )
        return model, training_time_seconds, warnings_list

    raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")


def evaluate_algorithm_on_validation(
    algorithm_name: str,
    model: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    # Avalia um algoritmo no conjunto de validacao.
    if algorithm_name == "mlp_classifier":
        return evaluate_mlp_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

    if algorithm_name == "rbf_network":
        return evaluate_rbf_network_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

    if algorithm_name == "fuzzy_cmeans":
        return evaluate_fuzzy_cmeans_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

    if algorithm_name == "sugeno_o0":
        return evaluate_zero_order_sugeno_classifier(
            model=model,
            X_validation=X_validation,
            y_validation=y_validation,
        )

    raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")


def run_single_algorithm(
    algorithm_name: str,
    prepared: dict[str, Any],
    seed: int,
    model_params: dict[str, Any] | None = None,
    search_iteration: int | None = None,
) -> dict[str, Any]:
    # Treina e avalia um algoritmo em um split ja preparado.
    X_train = prepared["processed_splits"]["X_train"]
    y_train = prepared["raw_splits"]["y_train"]
    X_validation = prepared["processed_splits"]["X_validation"]
    y_validation = prepared["raw_splits"]["y_validation"]
    n_classes = int(y_train.nunique())
    resolved_model_params = model_params or get_default_algorithm_params(
        algorithm_name=algorithm_name,
        n_classes=n_classes,
        seed=seed,
    )
    model, training_time_seconds, warnings_list = fit_algorithm_model(
        algorithm_name=algorithm_name,
        X_train=X_train,
        y_train=y_train,
        model_params=resolved_model_params,
    )
    evaluation = evaluate_algorithm_on_validation(
        algorithm_name=algorithm_name,
        model=model,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    split_signature = build_split_signature(prepared["split_indices"])
    return {
        "dataset_name": prepared["dataset_name"],
        "display_name": prepared["display_name"],
        "algorithm_name": algorithm_name,
        "algorithm_display_name": get_algorithm_display_name(algorithm_name),
        "seed": seed,
        "search_iteration": search_iteration,
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
        "model_params": serialize_model_params(resolved_model_params),
    }


def choose_best_record(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    # Seleciona a melhor configuracao pela acuracia de validacao com desempate deterministico.
    if not candidate_records:
        raise ValueError("Nao ha candidatos para selecionar a melhor configuracao.")

    sorted_records = sorted(
        candidate_records,
        key=lambda record: (
            -record["validation_accuracy"],
            record["training_time_seconds"],
            record["model_params"],
        ),
    )
    return dict(sorted_records[0])


def search_best_model_params_for_seed(
    prepared: dict[str, Any],
    seed: int,
    algorithm_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    # Executa busca em grade no conjunto de validacao e retorna os melhores parametros.
    y_train = prepared["raw_splits"]["y_train"]
    n_classes = int(y_train.nunique())
    selected_algorithms = algorithm_names or list(ALGORITHM_NAMES)

    all_candidate_records: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    best_params_by_algorithm: dict[str, dict[str, Any]] = {}

    for algorithm_name in selected_algorithms:
        param_grid = get_algorithm_param_grid(
            algorithm_name=algorithm_name,
            n_classes=n_classes,
            seed=seed,
        )
        candidate_records = []
        for search_iteration, model_params in enumerate(param_grid, start=1):
            candidate_record = run_single_algorithm(
                algorithm_name=algorithm_name,
                prepared=prepared,
                seed=seed,
                model_params=model_params,
                search_iteration=search_iteration,
            )
            candidate_records.append(candidate_record)
            all_candidate_records.append(candidate_record)

        best_record = choose_best_record(candidate_records)
        best_record["selection_metric"] = "validation_accuracy"
        best_record["grid_size"] = len(param_grid)
        best_records.append(best_record)
        best_params_by_algorithm[algorithm_name] = deserialize_model_params(
            best_record["model_params"]
        )

    return all_candidate_records, best_records, best_params_by_algorithm


def run_dataset_experiments_for_seed(
    dataset_name: str,
    seed: int,
    algorithm_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Executa a busca em grade e retorna candidatos e melhores configuracoes de uma seed.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    search_records, best_records, _ = search_best_model_params_for_seed(
        prepared=prepared,
        seed=seed,
        algorithm_names=algorithm_names,
    )
    return search_records, best_records


def build_runs_dataframe(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    # Converte as execucoes individuais em DataFrame tabular.
    return pd.DataFrame(run_records)


def build_summary_dataframe(runs_df: pd.DataFrame) -> pd.DataFrame:
    # Consolida media e desvio-padrao por dataset e algoritmo.
    summary_df = (
        runs_df.groupby(
            ["dataset_name", "display_name", "algorithm_name", "algorithm_display_name"],
            as_index=False,
        )
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
    search_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict[str, str]:
    # Salva as tabelas principais da ETAPA 8.
    ensure_experiments_directory()
    search_path = EXPERIMENTS_DIR / "validation_search_runs.csv"
    runs_path = EXPERIMENTS_DIR / "validation_runs.csv"
    summary_path = EXPERIMENTS_DIR / "validation_summary.csv"
    search_df.to_csv(search_path, index=False)
    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return {
        "search_path": str(search_path),
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

    search_records: list[dict[str, Any]] = []
    best_run_records: list[dict[str, Any]] = []
    for dataset_name in selected_datasets:
        for seed in seeds:
            dataset_search_records, dataset_best_records = run_dataset_experiments_for_seed(
                dataset_name=dataset_name,
                seed=seed,
                algorithm_names=algorithm_names,
            )
            search_records.extend(dataset_search_records)
            best_run_records.extend(dataset_best_records)

    search_df = build_runs_dataframe(search_records)
    runs_df = build_runs_dataframe(best_run_records)
    summary_df = build_summary_dataframe(runs_df)

    artifact_paths = {}
    if save_results:
        artifact_paths = save_experiment_results(
            search_df=search_df,
            runs_df=runs_df,
            summary_df=summary_df,
        )

    return {
        "dataset_names": selected_datasets,
        "seeds": seeds,
        "algorithm_names": algorithm_names or list(ALGORITHM_NAMES),
        "search_df": search_df,
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
        "Algoritmos: "
        + str(
            [
                get_algorithm_display_name(algorithm_name)
                for algorithm_name in experiment_result["algorithm_names"]
            ]
        ),
    ]

    if experiment_result["artifact_paths"]:
        lines.append(
            "Busca em grade salva em: "
            f"{experiment_result['artifact_paths']['search_path']}"
        )
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
            f"- {row.display_name} | {row.algorithm_display_name}: "
            f"acc_val = {row.validation_accuracy_mean:.4f} +- "
            f"{row.validation_accuracy_std:.4f}, "
            f"tempo = {row.training_time_mean_seconds:.4f} +- "
            f"{row.training_time_std_seconds:.4f} s"
        )

    return "\n".join(lines)
