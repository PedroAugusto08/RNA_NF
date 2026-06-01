from __future__ import annotations

import argparse

from src.config import DATASET_CONFIGS
from src.data_loader import format_dataset_report
from src.eda import format_eda_report, run_eda
from src.models import format_mlp_smoke_test_report, run_mlp_smoke_test
from src.preprocessing import (
    build_preprocessing_overview,
    format_preprocessing_report,
    prepare_all_datasets,
    prepare_dataset_splits,
)


def build_parser() -> argparse.ArgumentParser:
    # Cria o parser de argumentos de linha de comando.
    parser = argparse.ArgumentParser(
        description="Ferramentas do projeto de Inteligencia Computacional."
    )
    parser.add_argument(
        "--inspect-datasets",
        action="store_true",
        help="Mostra shape, colunas, tipos e primeiras linhas dos datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Inspeciona apenas um dataset especifico.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Quantidade de linhas mostradas no preview.",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Executa a analise exploratoria basica e salva tabelas e figuras.",
    )
    parser.add_argument(
        "--run-preprocessing",
        action="store_true",
        help="Executa divisao estratificada e preprocessamento reprodutivel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed usada na divisao treino/validacao/teste.",
    )
    parser.add_argument(
        "--test-mlp-classifier",
        action="store_true",
        help="Executa um teste rapido do MLP usando apenas treino e validacao.",
    )
    return parser


def main() -> None:
    # Executa a acao solicitada na linha de comando.
    parser = build_parser()
    args = parser.parse_args()

    if args.inspect_datasets:
        dataset_names = [args.dataset] if args.dataset else list(DATASET_CONFIGS)
        for index, dataset_name in enumerate(dataset_names, start=1):
            if index > 1:
                print("\n" + "=" * 80)
            print(format_dataset_report(dataset_name, n_rows=args.head))
        return

    if args.run_eda:
        dataset_names = [args.dataset] if args.dataset else list(DATASET_CONFIGS)
        summary_df = run_eda(dataset_names=dataset_names)
        print(format_eda_report(summary_df))
        return

    if args.run_preprocessing:
        if args.dataset:
            prepared = prepare_dataset_splits(dataset_name=args.dataset, seed=args.seed)
            summary_df = build_preprocessing_overview(
                {prepared["dataset_name"]: prepared}
            )
        else:
            prepared_all = prepare_all_datasets(seed=args.seed)
            summary_df = build_preprocessing_overview(prepared_all)

        print(format_preprocessing_report(summary_df))
        return

    if args.test_mlp_classifier:
        if not args.dataset:
            parser.error("Use --dataset ao executar --test-mlp-classifier.")

        result = run_mlp_smoke_test(dataset_name=args.dataset, seed=args.seed)
        print(format_mlp_smoke_test_report(result))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
