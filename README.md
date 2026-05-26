# Projeto IC - Comparacao de Modelos para Classificacao

Projeto da disciplina de Inteligencia Computacional com foco em comparacao metodologicamente correta de algoritmos para tarefas de classificacao em quatro datasets reais.

## Objetivo

Comparar modelos de redes neurais artificiais e modelos fuzzy/neuro-fuzzy usando uma metodologia experimental reprodutivel, com:

- analise exploratoria dos datasets;
- separacao treino/validacao/teste;
- busca sistematica de hiperparametros;
- multiplas execucoes independentes;
- metricas e analise estatistica;
- organizacao clara do codigo e dos resultados.

## Estrutura do projeto

```text
RNA_NF/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── results/
│   ├── figures/
│   ├── logs/
│   └── tables/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── experiments.py
│   ├── main.py
│   ├── preprocessing.py
│   └── models/
│       ├── __init__.py
│       ├── fuzzy_cmeans.py
│       ├── mlp_models.py
│       └── neuro_fuzzy.py
└── requirements.txt
```

## Organizacao dos datasets

Os arquivos CSV originais devem ficar em `data/raw/`.

Nomes esperados para manter o projeto padronizado:

- `breast_cancer.csv`
- `diabetes.csv`
- `faults.csv`
- `robot_navigation.csv`

Observacoes importantes:

- mantenha os arquivos originais sem preprocessamento em `data/raw/`;
- qualquer versao tratada ou transformada deve ir para `data/processed/`;
- se hoje seus arquivos estiverem com nomes diferentes, podemos padronizar isso na ETAPA 2 ou renomear manualmente antes de testar o carregamento.

## Como preparar o ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Escopo atual

Esta versao contem apenas a ETAPA 1:

- estrutura inicial de pastas;
- arquivos-base em `src/`;
- `requirements.txt`;
- `README.md` inicial.

Os modelos e a logica experimental ainda nao foram implementados.
