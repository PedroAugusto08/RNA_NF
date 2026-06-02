from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import (
    DATASET_CONFIGS,
    DEFAULT_EXPERIMENT_RUNS,
    DEFAULT_RANDOM_SEED,
    FIGURES_DIR,
    TABLES_DIR,
)
from src.experiments import (
    ALGORITHM_NAMES,
    fit_algorithm_model,
    generate_seeds,
    get_default_algorithm_params,
    serialize_model_params,
)
from src.preprocessing import prepare_dataset_splits


def ensure_evaluation_directories() -> None:
    # Garante que as pastas de tabelas e figuras existam.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_labels: list[Any],
) -> dict[str, Any]:
    # Calcula metricas de classificacao para binario e multiclasse.
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_macro": float(
            precision_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true_array,
                y_pred_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true_array,
                y_pred_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "positive_label": None,
        "precision_binary": np.nan,
        "recall_binary": np.nan,
        "f1_binary": np.nan,
    }

    if len(class_labels) == 2:
        positive_label = class_labels[-1]
        metrics["positive_label"] = str(positive_label)
        metrics["precision_binary"] = float(
            precision_score(
                y_true_array,
                y_pred_array,
                average="binary",
                pos_label=positive_label,
                zero_division=0,
            )
        )
        metrics["recall_binary"] = float(
            recall_score(
                y_true_array,
                y_pred_array,
                average="binary",
                pos_label=positive_label,
                zero_division=0,
            )
        )
        metrics["f1_binary"] = float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="binary",
                pos_label=positive_label,
                zero_division=0,
            )
        )

    return metrics


def save_confusion_matrix_figure(
    dataset_name: str,
    display_name: str,
    algorithm_name: str,
    seed: int,
    class_labels: list[Any],
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> str:
    # Salva a matriz de confusao em arquivo PNG.
    labels_as_str = [str(label) for label in class_labels]
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    output_path = (
        FIGURES_DIR
        / f"{dataset_name}_{algorithm_name}_seed_{seed}_confusion_matrix.png"
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"Matriz de Confusao - {display_name} - {algorithm_name}")
    plt.colorbar()
    tick_positions = np.arange(len(labels_as_str))
    plt.xticks(tick_positions, labels_as_str, rotation=45, ha="right")
    plt.yticks(tick_positions, labels_as_str)
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            text_color = "white" if value > threshold else "black"
            plt.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(output_path)


def evaluate_single_algorithm_on_test(
    algorithm_name: str,
    prepared: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    # Treina um algoritmo no treino e avalia no conjunto de teste.
    X_train = prepared["processed_splits"]["X_train"]
    y_train = prepared["raw_splits"]["y_train"]
    X_test = prepared["processed_splits"]["X_test"]
    y_test = prepared["raw_splits"]["y_test"]
    class_labels = list(np.unique(y_train))
    model_params = get_default_algorithm_params(
        algorithm_name=algorithm_name,
        n_classes=int(y_train.nunique()),
        seed=seed,
    )

    model, training_time_seconds, warnings_list = fit_algorithm_model(
        algorithm_name=algorithm_name,
        X_train=X_train,
        y_train=y_train,
        model_params=model_params,
    )
    y_pred = model.predict(X_test)
    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        class_labels=class_labels,
    )
    confusion_matrix_path = save_confusion_matrix_figure(
        dataset_name=prepared["dataset_name"],
        display_name=prepared["display_name"],
        algorithm_name=algorithm_name,
        seed=seed,
        class_labels=class_labels,
        y_true=y_test,
        y_pred=y_pred,
    )

    return {
        "dataset_name": prepared["dataset_name"],
        "display_name": prepared["display_name"],
        "algorithm_name": algorithm_name,
        "seed": seed,
        "n_classes": int(y_train.nunique()),
        "test_samples": int(X_test.shape[0]),
        "test_features": int(X_test.shape[1]),
        "training_time_seconds": float(training_time_seconds),
        "warning_count": len(warnings_list),
        "warnings": " | ".join(warnings_list),
        "model_params": serialize_model_params(model_params),
        "confusion_matrix_figure": confusion_matrix_path,
        **metrics,
    }


def run_evaluation_for_seed(
    dataset_name: str,
    seed: int,
    algorithm_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    # Avalia todos os algoritmos no mesmo split de teste para uma seed.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    selected_algorithms = algorithm_names or list(ALGORITHM_NAMES)
    return [
        evaluate_single_algorithm_on_test(
            algorithm_name=algorithm_name,
            prepared=prepared,
            seed=seed,
        )
        for algorithm_name in selected_algorithms
    ]


def build_metrics_by_run_dataframe(metric_records: list[dict[str, Any]]) -> pd.DataFrame:
    # Converte as avaliacoes individuais em DataFrame.
    return pd.DataFrame(metric_records)


def build_metrics_summary_dataframe(metrics_by_run_df: pd.DataFrame) -> pd.DataFrame:
    # Consolida media e desvio-padrao das metricas por dataset e algoritmo.
    summary_df = (
        metrics_by_run_df.groupby(
            ["dataset_name", "display_name", "algorithm_name"],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_binary_mean=("precision_binary", "mean"),
            precision_binary_std=("precision_binary", "std"),
            recall_binary_mean=("recall_binary", "mean"),
            recall_binary_std=("recall_binary", "std"),
            f1_binary_mean=("f1_binary", "mean"),
            f1_binary_std=("f1_binary", "std"),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", "std"),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            precision_weighted_mean=("precision_weighted", "mean"),
            precision_weighted_std=("precision_weighted", "std"),
            recall_weighted_mean=("recall_weighted", "mean"),
            recall_weighted_std=("recall_weighted", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=("f1_weighted", "std"),
            training_time_mean_seconds=("training_time_seconds", "mean"),
            training_time_std_seconds=("training_time_seconds", "std"),
            test_samples=("test_samples", "first"),
            test_features=("test_features", "first"),
            n_classes=("n_classes", "first"),
        )
    )
    std_columns = [
        "accuracy_std",
        "precision_binary_std",
        "recall_binary_std",
        "f1_binary_std",
        "precision_macro_std",
        "recall_macro_std",
        "f1_macro_std",
        "precision_weighted_std",
        "recall_weighted_std",
        "f1_weighted_std",
        "training_time_std_seconds",
    ]
    summary_df[std_columns] = summary_df[std_columns].fillna(0.0)
    return summary_df.sort_values(
        by=["dataset_name", "accuracy_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)


def save_metrics_results(
    metrics_by_run_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
) -> dict[str, str]:
    # Salva os CSVs principais da ETAPA 9.
    ensure_evaluation_directories()
    metrics_by_run_path = TABLES_DIR / "metrics_by_run.csv"
    metrics_summary_path = TABLES_DIR / "metrics_summary.csv"
    metrics_by_run_df.to_csv(metrics_by_run_path, index=False)
    metrics_summary_df.to_csv(metrics_summary_path, index=False)
    return {
        "metrics_by_run_path": str(metrics_by_run_path),
        "metrics_summary_path": str(metrics_summary_path),
    }


def run_evaluation(
    dataset_names: list[str] | None = None,
    n_runs: int = DEFAULT_EXPERIMENT_RUNS,
    start_seed: int = DEFAULT_RANDOM_SEED,
    algorithm_names: list[str] | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    # Executa a avaliacao completa no conjunto de teste para multiplas seeds.
    ensure_evaluation_directories()
    selected_datasets = dataset_names or list(DATASET_CONFIGS)
    seeds = generate_seeds(n_runs=n_runs, start_seed=start_seed)

    metric_records: list[dict[str, Any]] = []
    for dataset_name in selected_datasets:
        for seed in seeds:
            metric_records.extend(
                run_evaluation_for_seed(
                    dataset_name=dataset_name,
                    seed=seed,
                    algorithm_names=algorithm_names,
                )
            )

    metrics_by_run_df = build_metrics_by_run_dataframe(metric_records)
    metrics_summary_df = build_metrics_summary_dataframe(metrics_by_run_df)

    artifact_paths = {}
    if save_results:
        artifact_paths = save_metrics_results(
            metrics_by_run_df=metrics_by_run_df,
            metrics_summary_df=metrics_summary_df,
        )

    return {
        "dataset_names": selected_datasets,
        "seeds": seeds,
        "algorithm_names": algorithm_names or list(ALGORITHM_NAMES),
        "metrics_by_run_df": metrics_by_run_df,
        "metrics_summary_df": metrics_summary_df,
        "artifact_paths": artifact_paths,
    }


def format_evaluation_report(evaluation_result: dict[str, Any]) -> str:
    # Formata um resumo textual curto da ETAPA 9.
    summary_df = evaluation_result["metrics_summary_df"]
    lines = [
        "Avaliacao concluida com sucesso.",
        f"Datasets: {evaluation_result['dataset_names']}",
        f"Seeds: {evaluation_result['seeds']}",
        f"Algoritmos: {evaluation_result['algorithm_names']}",
    ]

    if evaluation_result["artifact_paths"]:
        lines.append(
            "Metricas por execucao salvas em: "
            f"{evaluation_result['artifact_paths']['metrics_by_run_path']}"
        )
        lines.append(
            "Resumo das metricas salvo em: "
            f"{evaluation_result['artifact_paths']['metrics_summary_path']}"
        )
        lines.append(f"Matrizes de confusao salvas em: {FIGURES_DIR}")

    lines.append("")
    lines.append("Resumo por dataset e algoritmo:")

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.display_name} | {row.algorithm_name}: "
            f"acc_teste = {row.accuracy_mean:.4f} +- {row.accuracy_std:.4f}, "
            f"f1_macro = {row.f1_macro_mean:.4f} +- {row.f1_macro_std:.4f}, "
            f"f1_weighted = {row.f1_weighted_mean:.4f} +- "
            f"{row.f1_weighted_std:.4f}"
        )

    return "\n".join(lines)
