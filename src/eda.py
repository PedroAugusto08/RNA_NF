from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import DATASET_CONFIGS, FIGURES_DIR, TABLES_DIR
from src.data_loader import load_dataset, read_raw_dataset


def ensure_results_directories() -> None:
    # Garante que as pastas de saida existam antes de salvar arquivos.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def build_column_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Monta uma tabela com nome da coluna e tipo detectado pelo pandas.
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    )


def build_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Resume valores ausentes por coluna.
    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / len(df) * 100).round(4)
    return pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_counts.values,
            "missing_percentage": missing_percent.values,
        }
    )


def build_class_distribution_summary(y: pd.Series) -> pd.DataFrame:
    # Resume a distribuicao da variavel alvo.
    class_counts = y.astype(str).value_counts().sort_index()
    class_percentages = (class_counts / len(y) * 100).round(4)
    return pd.DataFrame(
        {
            "class_label": class_counts.index,
            "count": class_counts.values,
            "percentage": class_percentages.values,
        }
    )


def save_class_distribution_plot(
    dataset_name: str,
    display_name: str,
    class_distribution_df: pd.DataFrame,
) -> Path:
    # Salva um grafico de barras com a distribuicao das classes.
    output_path = FIGURES_DIR / f"{dataset_name}_class_distribution.png"

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        class_distribution_df["class_label"],
        class_distribution_df["count"],
        color=plt.cm.Set2.colors[: len(class_distribution_df)],
    )
    ax = plt.gca()
    ax.set_title(f"Distribuicao das Classes - {display_name}")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade")
    plt.xticks(rotation=20, ha="right")

    for bar, count in zip(bars, class_distribution_df["count"]):
        ax.annotate(
            f"{int(count)}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def summarize_dataset(dataset_name: str) -> dict[str, Any]:
    # Gera o resumo principal e salva artefatos tabulares e graficos de um dataset.
    raw_df = read_raw_dataset(dataset_name)
    X, y, metadata = load_dataset(dataset_name)

    column_types_df = build_column_types_summary(raw_df)
    missing_values_df = build_missing_values_summary(raw_df)
    class_distribution_df = build_class_distribution_summary(y)

    column_types_path = TABLES_DIR / f"{metadata['dataset_name']}_column_types.csv"
    missing_values_path = TABLES_DIR / f"{metadata['dataset_name']}_missing_values.csv"
    class_distribution_path = (
        TABLES_DIR / f"{metadata['dataset_name']}_class_distribution.csv"
    )

    column_types_df.to_csv(column_types_path, index=False)
    missing_values_df.to_csv(missing_values_path, index=False)
    class_distribution_df.to_csv(class_distribution_path, index=False)

    figure_path = save_class_distribution_plot(
        dataset_name=metadata["dataset_name"],
        display_name=metadata["display_name"],
        class_distribution_df=class_distribution_df,
    )

    n_numeric_columns = int(raw_df.select_dtypes(include="number").shape[1])
    n_categorical_columns = int(raw_df.select_dtypes(exclude="number").shape[1])
    total_missing_values = int(raw_df.isna().sum().sum())
    columns_with_missing = int((raw_df.isna().sum() > 0).sum())

    return {
        "dataset_name": metadata["dataset_name"],
        "display_name": metadata["display_name"],
        "file_name": metadata["file_name"],
        "task_type": metadata["task_type"],
        "target_column": metadata["target_column"],
        "n_samples": int(raw_df.shape[0]),
        "n_total_columns_raw": int(raw_df.shape[1]),
        "n_features": int(X.shape[1]),
        "n_numeric_columns": n_numeric_columns,
        "n_categorical_columns": n_categorical_columns,
        "n_classes": metadata["n_classes"],
        "class_labels": " | ".join(metadata["class_labels"]),
        "total_missing_values": total_missing_values,
        "columns_with_missing": columns_with_missing,
        "class_distribution": " | ".join(
            f"{row.class_label}:{int(row.count)}"
            for row in class_distribution_df.itertuples(index=False)
        ),
        "column_types_table": str(column_types_path),
        "missing_values_table": str(missing_values_path),
        "class_distribution_table": str(class_distribution_path),
        "class_distribution_figure": str(figure_path),
    }


def run_eda(dataset_names: list[str] | None = None) -> pd.DataFrame:
    # Executa a EDA para um ou mais datasets e salva a tabela resumo consolidada.
    ensure_results_directories()
    selected_names = dataset_names or list(DATASET_CONFIGS)
    summary_rows = [summarize_dataset(dataset_name) for dataset_name in selected_names]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "datasets_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def format_eda_report(summary_df: pd.DataFrame) -> str:
    # Formata um resumo textual curto da EDA para exibicao no terminal.
    lines = [
        "EDA concluida com sucesso.",
        f"Tabela resumo salva em: {TABLES_DIR / 'datasets_summary.csv'}",
        f"Graficos salvos em: {FIGURES_DIR}",
        "",
        "Resumo dos datasets:",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.display_name}: {row.n_samples} amostras, "
            f"{row.n_features} atributos, {row.n_classes} classes, "
            f"ausentes totais = {row.total_missing_values}"
        )

    return "\n".join(lines)
