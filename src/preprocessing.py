from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATASET_CONFIGS,
    DEFAULT_RANDOM_SEED,
    PREPROCESSING_DIR,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)
from src.data_loader import load_dataset, normalize_dataset_name


def ensure_preprocessing_directory() -> None:
    # Garante que a pasta de artefatos de preprocessamento exista.
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)


def validate_split_sizes(
    train_size: float,
    validation_size: float,
    test_size: float,
) -> None:
    # Valida se as proporcoes informadas somam 1.0.
    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            "As proporcoes de treino, validacao e teste devem somar 1.0. "
            f"Valor atual: {total:.4f}"
        )


def detect_feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    # Separa colunas numericas e categoricas para o preprocessador.
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in X.columns if column not in numeric_columns
    ]
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    # Monta o preprocessador usando apenas as colunas observadas no treino.
    feature_groups = detect_feature_groups(X_train)
    transformers = []

    if feature_groups["numeric_columns"]:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, feature_groups["numeric_columns"])
        )

    if feature_groups["categorical_columns"]:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                feature_groups["categorical_columns"],
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, feature_groups


def create_split_indices(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    train_size: float = TRAIN_SIZE,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
) -> dict[str, pd.Index]:
    # Cria indices estratificados 60/20/20 reprodutiveis para reutilizacao futura.
    validate_split_sizes(train_size, validation_size, test_size)

    all_indices = X.index.to_numpy()
    train_indices, temp_indices = train_test_split(
        all_indices,
        train_size=train_size,
        random_state=seed,
        stratify=y.loc[all_indices],
    )

    validation_fraction_within_temp = validation_size / (validation_size + test_size)
    validation_indices, test_indices = train_test_split(
        temp_indices,
        train_size=validation_fraction_within_temp,
        random_state=seed,
        stratify=y.loc[temp_indices],
    )

    return {
        "train": pd.Index(sorted(train_indices)),
        "validation": pd.Index(sorted(validation_indices)),
        "test": pd.Index(sorted(test_indices)),
    }


def split_dataset_by_indices(
    X: pd.DataFrame,
    y: pd.Series,
    split_indices: dict[str, pd.Index],
) -> dict[str, Any]:
    # Aplica os indices gerados e retorna subconjuntos brutos.
    return {
        "X_train": X.loc[split_indices["train"]].copy(),
        "X_validation": X.loc[split_indices["validation"]].copy(),
        "X_test": X.loc[split_indices["test"]].copy(),
        "y_train": y.loc[split_indices["train"]].copy(),
        "y_validation": y.loc[split_indices["validation"]].copy(),
        "y_test": y.loc[split_indices["test"]].copy(),
    }


def transform_feature_splits(
    preprocessor: ColumnTransformer,
    raw_splits: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    # Ajusta no treino e transforma treino, validacao e teste com o mesmo preprocessador.
    preprocessor.fit(raw_splits["X_train"])
    feature_names = preprocessor.get_feature_names_out().tolist()

    transformed_splits = {}
    for split_name in ("train", "validation", "test"):
        X_key = f"X_{split_name}"
        transformed_array = preprocessor.transform(raw_splits[X_key])
        transformed_splits[X_key] = pd.DataFrame(
            transformed_array,
            index=raw_splits[X_key].index,
            columns=feature_names,
        )

    return transformed_splits, feature_names


def format_class_distribution(y_split: pd.Series) -> str:
    # Formata a distribuicao de classes para salvar em tabelas resumo.
    counts = y_split.astype(str).value_counts().sort_index()
    return " | ".join(f"{label}:{count}" for label, count in counts.items())


def save_split_assignments(
    dataset_name: str,
    seed: int,
    split_indices: dict[str, pd.Index],
) -> str:
    # Salva um arquivo com o papel de cada indice original no particionamento.
    rows = []
    for split_name, indices in split_indices.items():
        for original_index in indices.tolist():
            rows.append(
                {
                    "original_index": int(original_index),
                    "split": split_name,
                    "seed": seed,
                    "dataset_name": dataset_name,
                }
            )

    split_df = pd.DataFrame(rows).sort_values(by=["split", "original_index"])
    output_path = PREPROCESSING_DIR / f"{dataset_name}_seed_{seed}_split_assignments.csv"
    split_df.to_csv(output_path, index=False)
    return str(output_path)


def save_split_summary(
    dataset_name: str,
    seed: int,
    raw_splits: dict[str, Any],
    processed_splits: dict[str, pd.DataFrame],
    feature_groups: dict[str, list[str]],
    feature_names: list[str],
) -> str:
    # Salva um resumo tabular com tamanhos, distribuicoes e dimensoes apos preprocessamento.
    summary_rows = []
    for split_name in ("train", "validation", "test"):
        X_key = f"X_{split_name}"
        y_key = f"y_{split_name}"
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "seed": seed,
                "split": split_name,
                "n_samples": int(raw_splits[X_key].shape[0]),
                "n_raw_features": int(raw_splits[X_key].shape[1]),
                "n_processed_features": int(processed_splits[X_key].shape[1]),
                "class_distribution": format_class_distribution(raw_splits[y_key]),
                "numeric_columns_count": len(feature_groups["numeric_columns"]),
                "categorical_columns_count": len(feature_groups["categorical_columns"]),
                "processed_feature_names_count": len(feature_names),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    output_path = PREPROCESSING_DIR / f"{dataset_name}_seed_{seed}_split_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return str(output_path)


def save_feature_names(dataset_name: str, seed: int, feature_names: list[str]) -> str:
    # Salva os nomes das features apos preprocessamento.
    feature_names_df = pd.DataFrame({"feature_name": feature_names})
    output_path = PREPROCESSING_DIR / f"{dataset_name}_seed_{seed}_feature_names.csv"
    feature_names_df.to_csv(output_path, index=False)
    return str(output_path)


def prepare_dataset_splits(
    dataset_name: str,
    seed: int = DEFAULT_RANDOM_SEED,
    train_size: float = TRAIN_SIZE,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    # Carrega, divide e preprocessa um dataset com particionamento estratificado.
    normalized_name = normalize_dataset_name(dataset_name)
    X, y, metadata = load_dataset(normalized_name)
    split_indices = create_split_indices(
        X=X,
        y=y,
        seed=seed,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
    )
    raw_splits = split_dataset_by_indices(X=X, y=y, split_indices=split_indices)
    preprocessor, feature_groups = build_preprocessor(raw_splits["X_train"])
    processed_splits, feature_names = transform_feature_splits(
        preprocessor=preprocessor,
        raw_splits=raw_splits,
    )

    artifact_paths = {}
    if save_artifacts:
        ensure_preprocessing_directory()
        artifact_paths["split_assignments"] = save_split_assignments(
            dataset_name=normalized_name,
            seed=seed,
            split_indices=split_indices,
        )
        artifact_paths["split_summary"] = save_split_summary(
            dataset_name=normalized_name,
            seed=seed,
            raw_splits=raw_splits,
            processed_splits=processed_splits,
            feature_groups=feature_groups,
            feature_names=feature_names,
        )
        artifact_paths["feature_names"] = save_feature_names(
            dataset_name=normalized_name,
            seed=seed,
            feature_names=feature_names,
        )

    return {
        "dataset_name": normalized_name,
        "display_name": metadata["display_name"],
        "seed": seed,
        "split_sizes": {
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
        },
        "feature_groups": feature_groups,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "split_indices": split_indices,
        "raw_splits": raw_splits,
        "processed_splits": processed_splits,
        "artifact_paths": artifact_paths,
    }


def prepare_all_datasets(
    seed: int = DEFAULT_RANDOM_SEED,
    save_artifacts: bool = True,
) -> dict[str, dict[str, Any]]:
    # Executa o preprocessamento de todos os datasets configurados.
    return {
        dataset_name: prepare_dataset_splits(
            dataset_name=dataset_name,
            seed=seed,
            save_artifacts=save_artifacts,
        )
        for dataset_name in DATASET_CONFIGS
    }


def build_preprocessing_overview(
    prepared_data: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    # Consolida um resumo compacto dos particionamentos para exibicao ou salvamento.
    rows = []
    for dataset_name, prepared in prepared_data.items():
        raw_splits = prepared["raw_splits"]
        processed_splits = prepared["processed_splits"]
        rows.append(
            {
                "dataset_name": dataset_name,
                "display_name": prepared["display_name"],
                "seed": prepared["seed"],
                "train_samples": int(raw_splits["X_train"].shape[0]),
                "validation_samples": int(raw_splits["X_validation"].shape[0]),
                "test_samples": int(raw_splits["X_test"].shape[0]),
                "raw_features": int(raw_splits["X_train"].shape[1]),
                "processed_features": int(processed_splits["X_train"].shape[1]),
                "numeric_columns": len(prepared["feature_groups"]["numeric_columns"]),
                "categorical_columns": len(
                    prepared["feature_groups"]["categorical_columns"]
                ),
            }
        )

    return pd.DataFrame(rows)


def format_preprocessing_report(summary_df: pd.DataFrame) -> str:
    # Formata um resumo textual curto dos artefatos gerados na etapa.
    lines = [
        "Preprocessamento concluido com sucesso.",
        f"Artefatos salvos em: {PREPROCESSING_DIR}",
        "",
        "Resumo dos particionamentos:",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.display_name}: treino={row.train_samples}, "
            f"validacao={row.validation_samples}, teste={row.test_samples}, "
            f"features brutas={row.raw_features}, "
            f"features processadas={row.processed_features}"
        )

    return "\n".join(lines)
