from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATASET_ALIASES, DATASET_CONFIGS, DATA_DIR


TARGET_CANDIDATES = ("target", "class", "label", "diagnosis", "outcome", "y")


def normalize_dataset_name(dataset_name: str) -> str:
    # Normaliza aliases para a chave interna usada no projeto.
    normalized = dataset_name.strip().lower().replace(" ", "_")
    return DATASET_ALIASES.get(normalized, normalized)


def get_dataset_config(dataset_name: str) -> dict[str, Any]:
    # Retorna a configuracao do dataset ou levanta erro se nao existir.
    normalized_name = normalize_dataset_name(dataset_name)
    if normalized_name not in DATASET_CONFIGS:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(
            f"Dataset desconhecido: '{dataset_name}'. Opcoes disponiveis: {available}."
        )
    return DATASET_CONFIGS[normalized_name]


def get_dataset_path(dataset_name: str) -> Path:
    # Monta o caminho do CSV a partir da configuracao.
    config = get_dataset_config(dataset_name)
    return DATA_DIR / config["filename"]


def read_raw_dataset(dataset_name: str) -> pd.DataFrame:
    # Le o CSV bruto do dataset.
    dataset_path = get_dataset_path(dataset_name)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado para '{dataset_name}': {dataset_path}"
        )
    return pd.read_csv(dataset_path)


def resolve_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    # Resolve a coluna alvo usando configuracao e fallback heuristico.
    config = get_dataset_config(dataset_name)

    if "one_hot_target_columns" in config:
        return config["target_column"]

    configured_target = config.get("target_column")
    if configured_target and configured_target in df.columns:
        return configured_target

    lower_map = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    raise ValueError(
        f"Nao foi possivel identificar a coluna alvo para '{dataset_name}'. "
        "Verifique a configuracao do dataset."
    )


def collapse_faults_one_hot_target(
    df: pd.DataFrame, dataset_name: str = "faults"
) -> tuple[pd.DataFrame, pd.Series]:
    # Converte as colunas one-hot de falha em uma unica coluna target.
    config = get_dataset_config(dataset_name)
    target_columns = config["one_hot_target_columns"]
    missing = [column for column in target_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "As seguintes colunas one-hot de falha nao foram encontradas: "
            + ", ".join(missing)
        )

    row_sums = df[target_columns].sum(axis=1)
    invalid_rows = df.index[row_sums != 1].tolist()
    if invalid_rows:
        first_invalid = invalid_rows[:10]
        raise ValueError(
            "O dataset de falhas possui linhas com codificacao one-hot invalida. "
            f"Primeiros indices problematicos: {first_invalid}"
        )

    y = df[target_columns].idxmax(axis=1).rename(config["target_column"])
    X = df.drop(columns=target_columns)
    return X, y


def build_metadata(
    dataset_name: str,
    raw_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    dropped_columns: list[str],
) -> dict[str, Any]:
    # Constroi metadados padronizados do dataset.
    config = get_dataset_config(dataset_name)
    return {
        "dataset_name": normalize_dataset_name(dataset_name),
        "display_name": config["display_name"],
        "file_name": config["filename"],
        "file_path": str(get_dataset_path(dataset_name)),
        "task_type": config["task_type"],
        "n_samples": int(raw_df.shape[0]),
        "n_total_columns_raw": int(raw_df.shape[1]),
        "n_features": int(X.shape[1]),
        "target_column": target_column,
        "target_source_columns": list(config.get("one_hot_target_columns", [target_column])),
        "feature_columns": list(X.columns),
        "original_columns": list(raw_df.columns),
        "dropped_columns": dropped_columns,
        "feature_dtypes": {column: str(dtype) for column, dtype in X.dtypes.items()},
        "target_dtype": str(y.dtype),
        "n_classes": int(y.nunique()),
        "class_labels": sorted(y.astype(str).unique().tolist()),
        "class_distribution": y.astype(str).value_counts().sort_index().to_dict(),
        "head_preview": raw_df.head().to_dict(orient="records"),
    }


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    # Retorna X, y e metadados padronizados para um dataset.
    normalized_name = normalize_dataset_name(dataset_name)
    config = get_dataset_config(normalized_name)
    raw_df = read_raw_dataset(normalized_name)
    dropped_columns = list(config.get("drop_columns", []))

    if "one_hot_target_columns" in config:
        X, y = collapse_faults_one_hot_target(raw_df, normalized_name)
        target_column = config["target_column"]
    else:
        target_column = resolve_target_column(raw_df, normalized_name)
        y = raw_df[target_column].copy()
        X = raw_df.drop(columns=[target_column])

    safe_drop_columns = [column for column in dropped_columns if column in X.columns]
    if safe_drop_columns:
        X = X.drop(columns=safe_drop_columns)

    metadata = build_metadata(
        dataset_name=normalized_name,
        raw_df=raw_df,
        X=X,
        y=y,
        target_column=target_column,
        dropped_columns=safe_drop_columns,
    )
    return X, y, metadata


def load_all_datasets() -> dict[str, tuple[pd.DataFrame, pd.Series, dict[str, Any]]]:
    # Carrega todos os datasets configurados no projeto.
    return {
        dataset_name: load_dataset(dataset_name)
        for dataset_name in DATASET_CONFIGS
    }


def inspect_dataset(dataset_name: str, n_rows: int = 5) -> dict[str, Any]:
    # Retorna um resumo do dataset bruto e do dataset preparado.
    raw_df = read_raw_dataset(dataset_name)
    X, y, metadata = load_dataset(dataset_name)
    return {
        "dataset_name": metadata["dataset_name"],
        "display_name": metadata["display_name"],
        "shape_raw": raw_df.shape,
        "shape_features": X.shape,
        "target_name": metadata["target_column"],
        "columns": list(raw_df.columns),
        "dtypes": {column: str(dtype) for column, dtype in raw_df.dtypes.items()},
        "head": raw_df.head(n_rows).to_dict(orient="records"),
        "dropped_columns": metadata["dropped_columns"],
        "n_classes": metadata["n_classes"],
        "class_distribution": metadata["class_distribution"],
    }


def format_dataset_report(dataset_name: str, n_rows: int = 5) -> str:
    # Formata um relatorio textual com shape, colunas, tipos e primeiras linhas.
    summary = inspect_dataset(dataset_name, n_rows=n_rows)
    lines = [
        f"Dataset: {summary['display_name']} ({summary['dataset_name']})",
        f"Shape bruto: {summary['shape_raw']}",
        f"Shape de X: {summary['shape_features']}",
        f"Alvo identificado: {summary['target_name']}",
        f"Colunas removidas de X: {summary['dropped_columns']}",
        "Colunas:",
        ", ".join(summary["columns"]),
        "Tipos:",
    ]

    for column, dtype in summary["dtypes"].items():
        lines.append(f"  - {column}: {dtype}")

    lines.append("Primeiras linhas:")
    head_df = pd.DataFrame(summary["head"])
    if head_df.empty:
        lines.append("  <dataset vazio>")
    else:
        lines.append(head_df.to_string(index=False))

    lines.append(f"Distribuicao de classes: {summary['class_distribution']}")
    return "\n".join(lines)
