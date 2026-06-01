from __future__ import annotations

import argparse

from src.config import DATASET_CONFIGS
from src.data_loader import format_dataset_report
from src.eda import format_eda_report, run_eda
from src.models import (
    evaluate_mlp_classifier,
    evaluate_rbf_network_classifier,
    format_mlp_smoke_test_report,
    run_mlp_smoke_test,
    train_mlp_classifier,
    train_rbf_network_classifier,
)
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
    parser.add_argument(
        "--test-rna-models",
        action="store_true",
        help="Executa um teste rapido com MLP e RBF usando treino e validacao.",
    )
    return parser


def run_rna_models_smoke_test(dataset_name: str, seed: int) -> str:
    # Executa um smoke test conjunto para os dois modelos de RNA desta etapa.
    prepared = prepare_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
        save_artifacts=False,
    )
    X_train = prepared["processed_splits"]["X_train"]
    y_train = prepared["raw_splits"]["y_train"]
    X_validation = prepared["processed_splits"]["X_validation"]
    y_validation = prepared["raw_splits"]["y_validation"]

    mlp_params = {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "random_state": seed,
    }
    rbf_params = {
        "n_centers": max(len(y_train.unique()) * 2, 2),
        "gamma": None,
        "random_state": seed,
        "max_iter": 300,
        "output_max_iter": 1000,
    }

    mlp_model, mlp_training_time, mlp_warnings = train_mlp_classifier(
        X_train=X_train,
        y_train=y_train,
        **mlp_params,
    )
    mlp_result = evaluate_mlp_classifier(
        model=mlp_model,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    rbf_model, rbf_training_time, rbf_warnings = train_rbf_network_classifier(
        X_train=X_train,
        y_train=y_train,
        **rbf_params,
    )
    rbf_result = evaluate_rbf_network_classifier(
        model=rbf_model,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    lines = [
        "Teste dos modelos RNA concluido com sucesso.",
        f"Dataset: {prepared['display_name']} ({prepared['dataset_name']})",
        f"Seed: {seed}",
        f"Classes: {sorted(y_train.astype(str).unique().tolist())}",
        (
            f"Shapes -> treino: {X_train.shape}, "
            f"validacao: {X_validation.shape}, "
            f"teste preservado: {prepared['processed_splits']['X_test'].shape}"
        ),
        "",
        "MLP:",
        f"  - acuracia de validacao: {mlp_result['validation_accuracy']:.4f}",
        f"  - tempo de treino: {mlp_training_time:.4f} s",
        f"  - shape de predict_proba: {mlp_result.get('probabilities_shape')}",
        f"  - parametros: {mlp_params}",
        "",
        "RBF:",
        f"  - acuracia de validacao: {rbf_result['validation_accuracy']:.4f}",
        f"  - tempo de treino: {rbf_training_time:.4f} s",
        f"  - shape de predict_proba: {rbf_result['probabilities_shape']}",
        f"  - gamma efetivo: {rbf_model.gamma_:.8f}",
        f"  - centros: {rbf_model.centers_.shape}",
        f"  - parametros: {rbf_params}",
    ]

    if mlp_warnings:
        lines.append(f"  - avisos MLP: {mlp_warnings}")

    if rbf_warnings:
        lines.append(f"  - avisos RBF: {rbf_warnings}")

    return "\n".join(lines)


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

    if args.test_rna_models:
        if not args.dataset:
            parser.error("Use --dataset ao executar --test-rna-models.")

        smoke_report = run_rna_models_smoke_test(
            dataset_name=args.dataset,
            seed=args.seed,
        )
        print(smoke_report)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
