from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import DATASET_CONFIGS, EDA_FIGURES_DIR, FIGURES_DIR, TABLES_DIR
from src.data_loader import load_dataset, read_raw_dataset


def ensure_results_directories() -> None:
    # Garante que as pastas de saida existam antes de salvar arquivos.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def remove_legacy_eda_figures() -> None:
    # Remove figuras antigas da EDA salvas fora da subpasta dedicada.
    legacy_patterns = [
        "*_class_distribution.png",
        "*_correlation_heatmap.png",
        "*_pca_2d.png",
    ]
    for legacy_pattern in legacy_patterns:
        for legacy_figure_path in FIGURES_DIR.glob(legacy_pattern):
            legacy_figure_path.unlink(missing_ok=True)
        for legacy_figure_path in EDA_FIGURES_DIR.glob(legacy_pattern):
            if "correlation_heatmap" in legacy_figure_path.name:
                legacy_figure_path.unlink(missing_ok=True)


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
    output_path = EDA_FIGURES_DIR / f"{dataset_name}_class_distribution.png"

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


def build_pca_input_matrix(X: pd.DataFrame) -> pd.DataFrame:
    # Prepara uma matriz numerica para PCA, convertendo categoricas em dummies se necessario.
    if X.empty:
        raise ValueError("Nao foi possivel gerar PCA para um conjunto sem atributos.")

    encoded_X = pd.get_dummies(X, drop_first=False)
    if encoded_X.empty:
        raise ValueError("Nao foi possivel gerar PCA apos codificacao dos atributos.")

    return encoded_X.astype(float)


def save_pca_2d_plot(
    dataset_name: str,
    display_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> Path:
    # Salva uma projecao PCA 2D colorida pela classe.
    output_path = EDA_FIGURES_DIR / f"{dataset_name}_pca_2d.png"
    pca_input = build_pca_input_matrix(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_input)
    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame(
        {
            "pc1": components[:, 0],
            "pc2": components[:, 1],
            "class_label": y.astype(str).to_numpy(),
        }
    )
    explained_variance = pca.explained_variance_ratio_ * 100
    class_labels = sorted(plot_df["class_label"].unique().tolist())
    colors = plt.cm.Set2.colors

    plt.figure(figsize=(10, 6))
    for class_index, class_label in enumerate(class_labels):
        class_mask = plot_df["class_label"] == class_label
        plt.scatter(
            plot_df.loc[class_mask, "pc1"],
            plot_df.loc[class_mask, "pc2"],
            label=class_label,
            alpha=0.75,
            s=35,
            color=colors[class_index % len(colors)],
            edgecolors="none",
        )

    plt.title(
        f"PCA 2D - {display_name}\n"
        f"PC1 = {explained_variance[0]:.2f}% | PC2 = {explained_variance[1]:.2f}%"
    )
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(title="Classe")
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
    pca_figure_path = save_pca_2d_plot(
        dataset_name=metadata["dataset_name"],
        display_name=metadata["display_name"],
        X=X,
        y=y,
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
        "pca_2d_figure": str(pca_figure_path),
    }


def run_eda(dataset_names: list[str] | None = None) -> pd.DataFrame:
    # Executa a EDA para um ou mais datasets e salva a tabela resumo consolidada.
    ensure_results_directories()
    remove_legacy_eda_figures()
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
        f"Graficos salvos em: {EDA_FIGURES_DIR}",
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
