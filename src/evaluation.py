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
    get_algorithm_display_name,
)
from src.experiments import (
    ALGORITHM_NAMES,
    fit_algorithm_model,
    generate_seeds,
    search_best_model_params_for_seed,
    serialize_model_params,
)
from src.preprocessing import prepare_dataset_splits


def ensure_evaluation_directories() -> None:
    # Garante que as pastas de tabelas e figuras existam.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def remove_legacy_confusion_matrix_figures() -> None:
    # Remove figuras antigas por seed para manter apenas as matrizes finais agregadas.
    for legacy_figure_path in FIGURES_DIR.glob("*_seed_*_confusion_matrix.png"):
        legacy_figure_path.unlink(missing_ok=True)


FINAL_COMPARISON_PLOTS = [
    {
        "metric_column": "accuracy_mean",
        "std_column": "accuracy_std",
        "title": "Comparacao de Acuracia Media por Dataset",
        "ylabel": "Acuracia media no teste",
        "filename": "accuracy_comparison.png",
    },
    {
        "metric_column": "f1_macro_mean",
        "std_column": "f1_macro_std",
        "title": "Comparacao de F1 Macro Medio por Dataset",
        "ylabel": "F1 macro medio no teste",
        "filename": "f1_macro_comparison.png",
    },
]


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
    class_labels: list[Any],
    matrix: np.ndarray,
    n_runs: int,
) -> str:
    # Salva a matriz de confusao agregada em arquivo PNG.
    labels_as_str = [str(label) for label in class_labels]
    output_path = FIGURES_DIR / f"{dataset_name}_{algorithm_name}_confusion_matrix.png"

    # Normalizando a matriz para porcentagens:
    matrix_float = matrix.astype("float")
    row_sums = matrix_float.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix_normalized = matrix_float / row_sums

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix_normalized, interpolation="nearest", cmap="Blues")
    algorithm_display_name = get_algorithm_display_name(algorithm_name)
    plt.title(
        f"Matriz de Confusao Agregada - {display_name} - {algorithm_display_name}\n"
        f"({n_runs} execucoes)"
    )
    
    plt.colorbar(format="%.2f")
    tick_positions = np.arange(len(labels_as_str))
    plt.xticks(tick_positions, labels_as_str, rotation=45, ha="right")
    plt.yticks(tick_positions, labels_as_str)
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")

    threshold = 50

    for row_index in range(matrix_normalized.shape[0]):
        for column_index in range(matrix_normalized.shape[1]):
            value = matrix_normalized[row_index, column_index] * 100
            text_color = "white" if value > threshold else "black"
            cell_text = f"{value:.1f}%"
            plt.text(
                column_index,
                row_index,
                cell_text,
                ha="center",
                va="center",
                color=text_color,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(output_path)


def build_prediction_payload(
    dataset_name: str,
    display_name: str,
    algorithm_name: str,
    class_labels: list[Any],
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    # Guarda apenas o necessario para montar matrizes de confusao finais agregadas.
    return {
        "dataset_name": dataset_name,
        "display_name": display_name,
        "algorithm_name": algorithm_name,
        "algorithm_display_name": get_algorithm_display_name(algorithm_name),
        "class_labels": np.asarray(class_labels, dtype=str).tolist(),
        "y_true": np.asarray(y_true, dtype=str).tolist(),
        "y_pred": np.asarray(y_pred, dtype=str).tolist(),
    }


def save_final_confusion_matrix_figures(
    prediction_payloads: list[dict[str, Any]],
) -> list[str]:
    # Agrega as predicoes de todas as execucoes e salva uma matriz final por dataset e algoritmo.
    grouped_payloads: dict[tuple[str, str, str], dict[str, Any]] = {}

    for payload in prediction_payloads:
        group_key = (
            payload["dataset_name"],
            payload["display_name"],
            payload["algorithm_name"],
        )
        class_labels = np.asarray(payload["class_labels"], dtype=str).tolist()
        y_true = np.asarray(payload["y_true"], dtype=str)
        y_pred = np.asarray(payload["y_pred"], dtype=str)
        matrix = confusion_matrix(y_true, y_pred, labels=class_labels)

        if group_key not in grouped_payloads:
            grouped_payloads[group_key] = {
                "dataset_name": payload["dataset_name"],
                "display_name": payload["display_name"],
                "algorithm_name": payload["algorithm_name"],
                "algorithm_display_name": payload["algorithm_display_name"],
                "class_labels": class_labels,
                "matrix": matrix.astype(int),
                "n_runs": 1,
            }
        else:
            grouped_payloads[group_key]["matrix"] += matrix.astype(int)
            grouped_payloads[group_key]["n_runs"] += 1

    figure_paths: list[str] = []
    for aggregated_payload in grouped_payloads.values():
        figure_paths.append(
            save_confusion_matrix_figure(
                dataset_name=aggregated_payload["dataset_name"],
                display_name=aggregated_payload["display_name"],
                algorithm_name=aggregated_payload["algorithm_name"],
                class_labels=aggregated_payload["class_labels"],
                matrix=aggregated_payload["matrix"],
                n_runs=aggregated_payload["n_runs"],
            )
        )

    return sorted(figure_paths)


def save_comparison_plot(
    metrics_summary_df: pd.DataFrame,
    metric_column: str,
    std_column: str,
    title: str,
    ylabel: str,
    filename: str,
) -> str:
    # Salva um grafico comparativo agregado por dataset para uma metrica final.
    dataset_order = list(DATASET_CONFIGS)
    selected_df = metrics_summary_df.loc[
        metrics_summary_df["dataset_name"].isin(dataset_order)
    ].copy()
    selected_df["dataset_name"] = pd.Categorical(
        selected_df["dataset_name"],
        categories=dataset_order,
        ordered=True,
    )
    selected_df = selected_df.sort_values(["dataset_name", "algorithm_name"])

    datasets_df = (
        selected_df[["dataset_name", "display_name"]]
        .drop_duplicates()
        .sort_values("dataset_name")
    )
    dataset_names = datasets_df["dataset_name"].tolist()
    display_names = datasets_df["display_name"].tolist()
    algorithm_names = [
        algorithm_name
        for algorithm_name in ALGORITHM_NAMES
        if algorithm_name in selected_df["algorithm_name"].unique()
    ]

    x_positions = np.arange(len(dataset_names))
    total_width = 0.8
    bar_width = total_width / max(len(algorithm_names), 1)
    color_map = plt.cm.Set2(np.linspace(0, 1, max(len(algorithm_names), 1)))

    plt.figure(figsize=(12, 6))
    for algorithm_index, algorithm_name in enumerate(algorithm_names):
        algorithm_df = (
            selected_df.loc[selected_df["algorithm_name"] == algorithm_name]
            .set_index("dataset_name")
            .reindex(dataset_names)
        )
        offsets = x_positions - total_width / 2 + bar_width / 2 + algorithm_index * bar_width
        plt.bar(
            offsets,
            algorithm_df[metric_column].to_numpy(),
            width=bar_width,
            yerr=algorithm_df[std_column].to_numpy(),
            capsize=4,
            label=get_algorithm_display_name(algorithm_name),
            color=color_map[algorithm_index],
            alpha=0.9,
        )

    plt.title(title)
    plt.xlabel("Dataset")
    plt.ylabel(ylabel)
    plt.xticks(x_positions, display_names, rotation=15, ha="right")
    plt.legend(title="Algoritmo")
    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(output_path)


def save_final_comparison_plots(metrics_summary_df: pd.DataFrame) -> list[str]:
    # Salva apenas os graficos comparativos finais realmente usados no relatorio.
    return [
        save_comparison_plot(
            metrics_summary_df=metrics_summary_df,
            metric_column=plot_config["metric_column"],
            std_column=plot_config["std_column"],
            title=plot_config["title"],
            ylabel=plot_config["ylabel"],
            filename=plot_config["filename"],
        )
        for plot_config in FINAL_COMPARISON_PLOTS
    ]


def evaluate_single_algorithm_on_test(
    algorithm_name: str,
    prepared: dict[str, Any],
    seed: int,
    model_params: dict[str, Any],
    selected_validation_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Treina um algoritmo no treino e avalia no conjunto de teste.
    X_train = prepared["processed_splits"]["X_train"]
    y_train = prepared["raw_splits"]["y_train"]
    X_test = prepared["processed_splits"]["X_test"]
    y_test = prepared["raw_splits"]["y_test"]
    class_labels = list(np.unique(y_train))

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
    prediction_payload = build_prediction_payload(
        dataset_name=prepared["dataset_name"],
        display_name=prepared["display_name"],
        algorithm_name=algorithm_name,
        class_labels=class_labels,
        y_true=y_test,
        y_pred=y_pred,
    )

    metric_record = {
        "dataset_name": prepared["dataset_name"],
        "display_name": prepared["display_name"],
        "algorithm_name": algorithm_name,
        "algorithm_display_name": get_algorithm_display_name(algorithm_name),
        "seed": seed,
        "n_classes": int(y_train.nunique()),
        "test_samples": int(X_test.shape[0]),
        "test_features": int(X_test.shape[1]),
        "training_time_seconds": float(training_time_seconds),
        "selected_validation_accuracy": float(
            selected_validation_record["validation_accuracy"]
        ),
        "selection_metric": selected_validation_record["selection_metric"],
        "grid_size": int(selected_validation_record["grid_size"]),
        "selected_search_iteration": int(selected_validation_record["search_iteration"]),
        "warning_count": len(warnings_list),
        "warnings": " | ".join(warnings_list),
        "model_params": serialize_model_params(model_params),
        **metrics,
    }
    return metric_record, prediction_payload


def run_evaluation_for_seed(
    dataset_name: str,
    seed: int,
    algorithm_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Avalia todos os algoritmos no mesmo split de teste para uma seed.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    selected_algorithms = algorithm_names or list(ALGORITHM_NAMES)
    _, best_records, best_params_by_algorithm = search_best_model_params_for_seed(
        prepared=prepared,
        seed=seed,
        algorithm_names=selected_algorithms,
    )
    best_records_by_algorithm = {
        record["algorithm_name"]: record for record in best_records
    }
    metric_records: list[dict[str, Any]] = []
    prediction_payloads: list[dict[str, Any]] = []

    for algorithm_name in selected_algorithms:
        metric_record, prediction_payload = evaluate_single_algorithm_on_test(
            algorithm_name=algorithm_name,
            prepared=prepared,
            seed=seed,
            model_params=best_params_by_algorithm[algorithm_name],
            selected_validation_record=best_records_by_algorithm[algorithm_name],
        )
        metric_records.append(metric_record)
        prediction_payloads.append(prediction_payload)

    return metric_records, prediction_payloads


def build_metrics_by_run_dataframe(metric_records: list[dict[str, Any]]) -> pd.DataFrame:
    # Converte as avaliacoes individuais em DataFrame.
    return pd.DataFrame(metric_records)


def build_metrics_summary_dataframe(metrics_by_run_df: pd.DataFrame) -> pd.DataFrame:
    # Consolida media e desvio-padrao das metricas por dataset e algoritmo.
    summary_df = (
        metrics_by_run_df.groupby(
            [
                "dataset_name",
                "display_name",
                "algorithm_name",
                "algorithm_display_name",
            ],
            as_index=False,
        )
        .agg(
            n_runs=("seed", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            selected_validation_accuracy_mean=("selected_validation_accuracy", "mean"),
            selected_validation_accuracy_std=("selected_validation_accuracy", "std"),
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
        "selected_validation_accuracy_std",
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
    prediction_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    # Salva os CSVs principais da ETAPA 9.
    ensure_evaluation_directories()
    remove_legacy_confusion_matrix_figures()
    metrics_by_run_path = TABLES_DIR / "metrics_by_run.csv"
    metrics_summary_path = TABLES_DIR / "metrics_summary.csv"
    metrics_by_run_df.to_csv(metrics_by_run_path, index=False)
    metrics_summary_df.to_csv(metrics_summary_path, index=False)
    confusion_matrix_paths = save_final_confusion_matrix_figures(prediction_payloads)
    comparison_plot_paths = save_final_comparison_plots(metrics_summary_df)
    return {
        "metrics_by_run_path": str(metrics_by_run_path),
        "metrics_summary_path": str(metrics_summary_path),
        "confusion_matrix_paths": confusion_matrix_paths,
        "comparison_plot_paths": comparison_plot_paths,
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
    prediction_payloads: list[dict[str, Any]] = []
    for dataset_name in selected_datasets:
        for seed in seeds:
            seed_metric_records, seed_prediction_payloads = run_evaluation_for_seed(
                dataset_name=dataset_name,
                seed=seed,
                algorithm_names=algorithm_names,
            )
            metric_records.extend(seed_metric_records)
            prediction_payloads.extend(seed_prediction_payloads)

    metrics_by_run_df = build_metrics_by_run_dataframe(metric_records)
    metrics_summary_df = build_metrics_summary_dataframe(metrics_by_run_df)

    artifact_paths = {}
    if save_results:
        artifact_paths = save_metrics_results(
            metrics_by_run_df=metrics_by_run_df,
            metrics_summary_df=metrics_summary_df,
            prediction_payloads=prediction_payloads,
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
        "Algoritmos: "
        + str(
            [
                get_algorithm_display_name(algorithm_name)
                for algorithm_name in evaluation_result["algorithm_names"]
            ]
        ),
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
        lines.append(f"Figuras finais salvas em: {FIGURES_DIR}")

    lines.append("")
    lines.append("Resumo por dataset e algoritmo:")

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.display_name} | {row.algorithm_display_name}:\n"
            f"  acc_val_sel = {row.selected_validation_accuracy_mean:.4f} +- {row.selected_validation_accuracy_std:.4f}\n"
            f"  acc_teste   = {row.accuracy_mean:.4f} +- {row.accuracy_std:.4f}\n"
            f"  precisao    = {row.precision_macro_mean:.4f} +- {row.precision_macro_std:.4f}\n"
            f"  revogacao   = {row.recall_macro_mean:.4f} +- {row.recall_macro_std:.4f}\n"
            f"  f1_macro    = {row.f1_macro_mean:.4f} +- {row.f1_macro_std:.4f}\n"
            f"  f1_weighted = {row.f1_weighted_mean:.4f} +- {row.f1_weighted_std:.4f}"
        )

    return "\n".join(lines)
