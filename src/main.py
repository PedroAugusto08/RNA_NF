from __future__ import annotations

import argparse

from src.config import DATASET_CONFIGS
from src.data_loader import format_dataset_report
from src.eda import format_eda_report, run_eda


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

    parser.print_help()


if __name__ == "__main__":
    main()
