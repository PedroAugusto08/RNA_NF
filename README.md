# Projeto IC - Comparação de Modelos para Classificação

Projeto da disciplina de Inteligência Computacional para comparação experimental de modelos de classificação em quatro bases reais, com foco em reprodutibilidade, seleção sistemática de hiperparâmetros e análise estatística dos resultados.

## Visão geral

O projeto compara quatro algoritmos:

- `MLP Classifier`
- `RBF Network`
- `Fuzzy C-Means Classifier`
- `Fuzzy KNN Classifier`

Os experimentos usam:

- quatro datasets reais de classificação;
- divisão estratificada em treino, validação e teste (`60/20/20`);
- busca em grade no conjunto de validação;
- `21` execuções independentes por padrão, com seeds diferentes;
- avaliação final no conjunto de teste;
- ranking e análise estatística dos algoritmos.

## Estrutura

```text
RNA_NF/
├── data/
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── experiments.py
│   ├── evaluation.py
│   ├── statistical_analysis.py
│   ├── main.py
│   └── models/
└── requirements.txt
```

## Datasets

Os arquivos CSV devem ficar em `data/`:

- `breast-cancer.csv`
- `diabetes.csv`
- `faults.csv`
- `Robo.csv`

## Execução

Criação do ambiente e instalação das dependências:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Comando principal de reprodução experimental:

```powershell
python -m src.main --run
```

Comando auxiliar para análise exploratória:

```powershell
python -m src.main --run-eda
```

## Saídas

Os principais artefatos gerados ficam em `results/`:

- tabelas com métricas por execução e métricas resumidas;
- tabelas de ranking e comparação estatística;
- figuras de distribuição de classes e matrizes de confusão.

O detalhamento metodológico, a discussão dos resultados e a interpretação final ficam reservados para o artigo.
