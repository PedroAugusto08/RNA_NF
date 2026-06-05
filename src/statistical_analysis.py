from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from src.config import TABLES_DIR


METRICS_FOR_STATISTICAL_ANALYSIS = [
    "accuracy_mean",
    "f1_macro_mean",
    "f1_weighted_mean",
]


def load_metrics_summary() -> pd.DataFrame:
    # Carrega o resumo de metricas da ETAPA 9.
    metrics_summary_path = TABLES_DIR / "metrics_summary.csv"
    if not metrics_summary_path.exists():
        raise FileNotFoundError(
            "Arquivo de metricas nao encontrado. Rode primeiro a ETAPA 9 com "
            "`python -m src.main --run-evaluation ...`."
        )
    return pd.read_csv(metrics_summary_path)


def build_rankings_by_dataset(
    metrics_summary_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    # Gera ranking por dataset para cada metrica escolhida.
    selected_metrics = metrics or list(METRICS_FOR_STATISTICAL_ANALYSIS)
    ranking_rows: list[dict[str, Any]] = []

    for metric_name in selected_metrics:
        for dataset_name, group_df in metrics_summary_df.groupby("dataset_name"):
            group_copy = group_df.copy()
            ranks = rankdata(-group_copy[metric_name].to_numpy(), method="average")
            group_copy["rank"] = ranks

            for row in group_copy.itertuples(index=False):
                ranking_rows.append(
                    {
                        "metric_name": metric_name,
                        "dataset_name": row.dataset_name,
                        "display_name": row.display_name,
                        "algorithm_name": row.algorithm_name,
                        "score": getattr(row, metric_name),
                        "rank": row.rank,
                    }
                )

    ranking_df = pd.DataFrame(ranking_rows)
    return ranking_df.sort_values(
        by=["metric_name", "dataset_name", "rank", "algorithm_name"]
    ).reset_index(drop=True)


def build_average_ranking(ranking_df: pd.DataFrame) -> pd.DataFrame:
    # Consolida ranking medio e score medio por algoritmo.
    average_ranking_df = (
        ranking_df.groupby(["metric_name", "algorithm_name"], as_index=False)
        .agg(
            average_rank=("rank", "mean"),
            rank_std=("rank", "std"),
            average_score=("score", "mean"),
            score_std=("score", "std"),
            n_datasets=("dataset_name", "nunique"),
        )
    )
    average_ranking_df["rank_std"] = average_ranking_df["rank_std"].fillna(0.0)
    average_ranking_df["score_std"] = average_ranking_df["score_std"].fillna(0.0)
    return average_ranking_df.sort_values(
        by=["metric_name", "average_rank", "algorithm_name"]
    ).reset_index(drop=True)


def run_friedman_analysis(
    metrics_summary_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    # Aplica Friedman usando as medias por dataset quando ha dados completos.
    selected_metrics = metrics or list(METRICS_FOR_STATISTICAL_ANALYSIS)
    friedman_rows: list[dict[str, Any]] = []

    for metric_name in selected_metrics:
        pivot_df = (
            metrics_summary_df.pivot(
                index="dataset_name",
                columns="algorithm_name",
                values=metric_name,
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        complete_pivot_df = pivot_df.dropna(axis=0, how="any")
        n_datasets = int(complete_pivot_df.shape[0])
        n_algorithms = int(complete_pivot_df.shape[1])

        if n_datasets < 2 or n_algorithms < 2:
            friedman_rows.append(
                {
                    "metric_name": metric_name,
                    "n_datasets": n_datasets,
                    "n_algorithms": n_algorithms,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "status": "insufficient_data",
                }
            )
            continue

        statistic, p_value = friedmanchisquare(
            *[complete_pivot_df[column].to_numpy() for column in complete_pivot_df.columns]
        )
        friedman_rows.append(
            {
                "metric_name": metric_name,
                "n_datasets": n_datasets,
                "n_algorithms": n_algorithms,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "status": "ok",
            }
        )

    return pd.DataFrame(friedman_rows)


def apply_holm_correction(p_values: list[float]) -> list[float]:
    # Aplica a correcao de Holm-Bonferroni a uma lista de p-valores.
    if not p_values:
        return []

    indexed_values = sorted(enumerate(p_values), key=lambda item: item[1])
    corrected = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)

    for rank_index, (original_index, p_value) in enumerate(indexed_values, start=1):
        adjusted = (total - rank_index + 1) * p_value
        adjusted = min(adjusted, 1.0)
        running_max = max(running_max, adjusted)
        corrected[original_index] = running_max

    return corrected


def run_pairwise_wilcoxon_posthoc(
    metrics_summary_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    metrics: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    # Executa comparacoes pareadas entre algoritmos como pos-teste simples.
    # A interpretacao final considera o Friedman global e a correcao de Holm.
    selected_metrics = metrics or list(METRICS_FOR_STATISTICAL_ANALYSIS)
    posthoc_rows: list[dict[str, Any]] = []

    for metric_name in selected_metrics:
        pivot_df = (
            metrics_summary_df.pivot(
                index="dataset_name",
                columns="algorithm_name",
                values=metric_name,
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        complete_pivot_df = pivot_df.dropna(axis=0, how="any")
        algorithm_names = complete_pivot_df.columns.tolist()
        metric_friedman = friedman_df.loc[
            friedman_df["metric_name"] == metric_name
        ].iloc[0]
        friedman_significant = bool(
            metric_friedman["status"] == "ok"
            and metric_friedman["p_value"] < alpha
        )
        raw_p_values: list[float] = []
        pair_records: list[dict[str, Any]] = []

        for algorithm_a, algorithm_b in combinations(algorithm_names, 2):
            scores_a = complete_pivot_df[algorithm_a].to_numpy()
            scores_b = complete_pivot_df[algorithm_b].to_numpy()
            differences = scores_a - scores_b

            if complete_pivot_df.shape[0] < 2 or np.allclose(differences, 0.0):
                statistic = np.nan
                p_value = 1.0
                status = "insufficient_variation"
            else:
                statistic, p_value = wilcoxon(
                    scores_a,
                    scores_b,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(statistic)
                p_value = float(p_value)
                status = "ok"

            raw_p_values.append(p_value)
            pair_records.append(
                {
                    "metric_name": metric_name,
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "n_datasets": int(complete_pivot_df.shape[0]),
                    "mean_score_a": float(scores_a.mean()),
                    "mean_score_b": float(scores_b.mean()),
                    "mean_difference_a_minus_b": float(differences.mean()),
                    "statistic": statistic,
                    "p_value_raw": p_value,
                    "friedman_p_value": float(metric_friedman["p_value"])
                    if metric_friedman["status"] == "ok"
                    else np.nan,
                    "friedman_significant": friedman_significant,
                    "status": status,
                }
            )

        corrected_p_values = apply_holm_correction(raw_p_values)
        for pair_record, corrected_p_value in zip(pair_records, corrected_p_values):
            pair_record["p_value_holm"] = corrected_p_value
            pair_record["significant_after_holm"] = bool(corrected_p_value < alpha)
            pair_record["significant_final"] = bool(
                friedman_significant and corrected_p_value < alpha
            )
            posthoc_rows.append(pair_record)

    return pd.DataFrame(posthoc_rows)


def build_statistical_comparison_table(
    average_ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    # Gera uma tabela comparativa final em formato wide.
    comparison_df = average_ranking_df.pivot(
        index="algorithm_name",
        columns="metric_name",
        values=["average_rank", "average_score"],
    )
    comparison_df.columns = [
        f"{outer}_{inner}" for outer, inner in comparison_df.columns.to_flat_index()
    ]
    comparison_df = comparison_df.reset_index()
    return comparison_df.sort_values(by="algorithm_name").reset_index(drop=True)


def save_statistical_analysis_results(
    rankings_by_dataset_df: pd.DataFrame,
    average_ranking_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    posthoc_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> dict[str, str]:
    # Salva os artefatos tabulares da ETAPA 10.
    rankings_path = TABLES_DIR / "rankings_by_dataset.csv"
    average_ranking_path = TABLES_DIR / "average_ranking.csv"
    friedman_path = TABLES_DIR / "friedman_results.csv"
    posthoc_path = TABLES_DIR / "pairwise_wilcoxon_results.csv"
    comparison_path = TABLES_DIR / "statistical_comparison.csv"

    rankings_by_dataset_df.to_csv(rankings_path, index=False)
    average_ranking_df.to_csv(average_ranking_path, index=False)
    friedman_df.to_csv(friedman_path, index=False)
    posthoc_df.to_csv(posthoc_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    return {
        "rankings_path": str(rankings_path),
        "average_ranking_path": str(average_ranking_path),
        "friedman_path": str(friedman_path),
        "posthoc_path": str(posthoc_path),
        "comparison_path": str(comparison_path),
    }


def run_statistical_analysis(
    metrics: list[str] | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    # Executa ranking, Friedman e pos-teste em cima do resumo da ETAPA 9.
    metrics_summary_df = load_metrics_summary()
    rankings_by_dataset_df = build_rankings_by_dataset(
        metrics_summary_df=metrics_summary_df,
        metrics=metrics,
    )
    average_ranking_df = build_average_ranking(rankings_by_dataset_df)
    friedman_df = run_friedman_analysis(
        metrics_summary_df=metrics_summary_df,
        metrics=metrics,
    )
    posthoc_df = run_pairwise_wilcoxon_posthoc(
        metrics_summary_df=metrics_summary_df,
        friedman_df=friedman_df,
        metrics=metrics,
    )
    comparison_df = build_statistical_comparison_table(average_ranking_df)

    artifact_paths = {}
    if save_results:
        artifact_paths = save_statistical_analysis_results(
            rankings_by_dataset_df=rankings_by_dataset_df,
            average_ranking_df=average_ranking_df,
            friedman_df=friedman_df,
            posthoc_df=posthoc_df,
            comparison_df=comparison_df,
        )

    return {
        "metrics_summary_df": metrics_summary_df,
        "rankings_by_dataset_df": rankings_by_dataset_df,
        "average_ranking_df": average_ranking_df,
        "friedman_df": friedman_df,
        "posthoc_df": posthoc_df,
        "comparison_df": comparison_df,
        "artifact_paths": artifact_paths,
    }


def format_statistical_analysis_report(result: dict[str, Any]) -> str:
    # Formata um resumo textual curto da ETAPA 10.
    average_ranking_df = result["average_ranking_df"]
    friedman_df = result["friedman_df"]
    posthoc_df = result["posthoc_df"]
    lines = ["Analise estatistica concluida com sucesso."]

    if result["artifact_paths"]:
        lines.append(
            f"Rankings por dataset salvos em: {result['artifact_paths']['rankings_path']}"
        )
        lines.append(
            "Ranking medio salvo em: "
            f"{result['artifact_paths']['average_ranking_path']}"
        )
        lines.append(
            f"Resultados de Friedman salvos em: {result['artifact_paths']['friedman_path']}"
        )
        lines.append(
            "Comparacoes pareadas salvas em: "
            f"{result['artifact_paths']['posthoc_path']}"
        )
        lines.append(
            "Tabela comparativa final salva em: "
            f"{result['artifact_paths']['comparison_path']}"
        )

    lines.append("")
    lines.append("Ranking medio por metrica:")
    for row in average_ranking_df.itertuples(index=False):
        lines.append(
            f"- {row.metric_name} | {row.algorithm_name}: "
            f"rank medio = {row.average_rank:.4f}, "
            f"score medio = {row.average_score:.4f}"
        )

    lines.append("")
    lines.append("Friedman:")
    for row in friedman_df.itertuples(index=False):
        if row.status != "ok":
            lines.append(
                f"- {row.metric_name}: dados insuficientes para o teste."
            )
        else:
            lines.append(
                f"- {row.metric_name}: estatistica = {row.statistic:.4f}, "
                f"p-valor = {row.p_value:.6f}"
            )

    significant_pairs_df = posthoc_df.loc[posthoc_df["significant_final"]].copy()
    lines.append("")
    lines.append("Pos-teste pareado (Wilcoxon + Holm):")
    if significant_pairs_df.empty:
        lines.append("- Nenhuma diferenca par-a-par permaneceu significativa apos Holm.")
    else:
        for row in significant_pairs_df.itertuples(index=False):
            lines.append(
                f"- {row.metric_name}: {row.algorithm_a} vs {row.algorithm_b} | "
                f"p_holm = {row.p_value_holm:.6f}"
            )

    return "\n".join(lines)
