# Projeto IC - Comparação de Modelos para Classificação

Projeto da disciplina de Inteligência Computacional para comparação experimental de algoritmos de classificação em quatro datasets reais. O foco principal do trabalho é a metodologia experimental: divisão correta dos dados, seleção sistemática de hiperparâmetros, repetição com múltiplas seeds, métricas adequadas e análise estatística dos resultados.

## Objetivo

O projeto compara quatro modelos de Inteligência Computacional:

- `MLP`
- `RBF Network`
- `Fuzzy C-Means`
- `Takagi-Sugeno de Ordem Zero`

A comparação é feita em quatro bases reais de classificação, com:

- divisão estratificada em treino, validação e teste (`60/20/20`);
- busca em grade no conjunto de validação;
- `21` execuções independentes por padrão;
- uso das mesmas seeds e dos mesmos splits para todos os algoritmos em cada rodada;
- avaliação final no conjunto de teste;
- análise estatística para comparação entre algoritmos.

## Datasets

Os arquivos CSV devem ficar em `data/`:

- `breast-cancer.csv`
- `diabetes.csv`
- `faults.csv`
- `Robo.csv`

As bases usadas são:

1. `Breast Cancer Wisconsin`
   Classificação binária.

2. `Pima Indians Diabetes`
   Classificação binária.

3. `Steel Plates Faults`
   Classificação multiclasse.

4. `Wall-Following Robot Navigation`
   Classificação multiclasse.

Observação importante:
- no dataset `faults.csv`, as 7 colunas one-hot de falha são convertidas automaticamente para uma única coluna alvo chamada `fault_type`.

## Modelos implementados

### 1. MLP

Rede neural treinada por backpropagation usando `sklearn.neural_network.MLPClassifier`.

Hiperparâmetros avaliados:
- número de neurônios/camadas ocultas;
- função de ativação;
- regularização `alpha`;
- taxa de aprendizado inicial.

### 2. RBF Network

Rede neural com camada de funções de base radial.

Implementação:
- centros obtidos por `KMeans`;
- ativações gaussianas com parâmetro `gamma`;
- camada de saída com `LogisticRegression`.

Hiperparâmetros avaliados:
- número de centros;
- `gamma`.

### 3. Fuzzy C-Means

Classificador fuzzy baseado em clustering.

Implementação:
- aplica `Fuzzy C-Means` no treino;
- associa clusters às classes;
- usa pertinências para produzir probabilidades e predições.

Hiperparâmetros avaliados:
- número de clusters;
- parâmetro fuzzy `m`.

### 4. Sugeno_O0

Sistema fuzzy do tipo Sugeno de ordem zero.

Implementação:
- regras geradas a partir de `KMeans`;
- antecedentes gaussianos;
- consequentes constantes por classe;
- saída final por agregação ponderada das regras.

Hiperparâmetros avaliados:
- número de regras;
- escala das larguras gaussianas (`sigma_scale`).

## Metodologia experimental

O fluxo experimental foi implementado da seguinte forma:

### Divisão dos dados

Cada dataset é dividido em:

- `60%` treino
- `20%` validação
- `20%` teste

A divisão é:

- estratificada;
- reprodutível por seed;
- compartilhada entre todos os algoritmos dentro da mesma execução.

### Pré-processamento

O pré-processamento é aplicado usando apenas o conjunto de treino como referência.

Etapas implementadas:

- separação treino/validação/teste;
- normalização de atributos numéricos;
- suporte a codificação de variáveis categóricas;
- suporte a imputação condicional, embora os datasets atuais não tenham valores ausentes.

### Seleção de hiperparâmetros

Cada algoritmo possui uma grade pequena de hiperparâmetros. A escolha da melhor configuração é feita:

- no conjunto de validação;
- de forma sistemática por busca em grade;
- sem uso do conjunto de teste nessa etapa.

### Execuções independentes

Por padrão, o projeto executa `21` rodadas independentes, com seeds diferentes.

Exemplo com `seed=42`:
- `42, 43, 44, ..., 62`

Isso permite medir:

- valor médio das métricas;
- desvio-padrão;
- estabilidade dos modelos.

## Métricas e análise estatística

As métricas calculadas na avaliação final incluem:

- `accuracy`
- `precision`
- `recall`
- `f1-score`

Para problemas multiclasse, o projeto salva métricas:

- `macro`
- `weighted`

Também são geradas:

- matrizes de confusão finais;
- ranking por dataset;
- ranking médio global;
- teste de `Friedman`;
- pós-teste pareado com `Wilcoxon + Holm`.

## Estrutura do projeto

```text
RNA_NF/
├── data/
│   ├── breast-cancer.csv
│   ├── diabetes.csv
│   ├── faults.csv
│   └── Robo.csv
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
│       ├── __init__.py
│       ├── mlp_classifier.py
│       ├── rbf_network.py
│       ├── fuzzy_cmeans.py
│       └── sugeno_zero_order.py
└── requirements.txt
```

## Papel dos principais arquivos

- `src/config.py`
  Configurações centrais do projeto, caminhos, datasets e nomes amigáveis dos algoritmos.

- `src/data_loader.py`
  Leitura dos CSVs, identificação do alvo e padronização dos datasets.

- `src/eda.py`
  Análise exploratória básica e geração das figuras/tabelas descritivas.

- `src/preprocessing.py`
  Splits estratificados, normalização e preparação reprodutível dos dados.

- `src/experiments.py`
  Busca em grade na validação e controle das execuções por seed.

- `src/evaluation.py`
  Avaliação final no teste, cálculo das métricas e geração das figuras finais.

- `src/statistical_analysis.py`
  Ranking e testes estatísticos.

- `src/main.py`
  Interface de linha de comando do projeto.

## Como executar

### 1. Criar ambiente e instalar dependências

No PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux/WSL:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Rodar o pipeline principal

```powershell
python -m src.main --run
```

Esse comando:

- roda todos os 4 datasets;
- roda todos os 4 algoritmos;
- executa `21` rodadas por padrão;
- faz seleção de hiperparâmetros na validação;
- avalia no teste;
- gera tabelas finais;
- executa a análise estatística.

### 3. Rodar apenas a EDA

```powershell
python -m src.main --run-eda
```

### 4. Rodar apenas um dataset

```powershell
python -m src.main --run --dataset breast_cancer
```

### 5. Ajustar número de execuções ou seed inicial

```powershell
python -m src.main --run --n-runs 21 --seed 42
```

## Saídas geradas

Os principais resultados ficam em `results/`.

### Tabelas

Exemplos de tabelas geradas:

- `datasets_summary.csv`
- `metrics_by_run.csv`
- `metrics_summary.csv`
- `rankings_by_dataset.csv`
- `average_ranking.csv`
- `friedman_results.csv`
- `pairwise_wilcoxon_results.csv`
- `statistical_comparison.csv`

### Figuras

O pipeline foi ajustado para gerar apenas figuras mais relevantes para o relatório:

- distribuição de classes por dataset;
- matrizes de confusão finais por dataset e algoritmo;
- gráfico comparativo de `accuracy`;
- gráfico comparativo de `f1_macro`.

## Comandos auxiliares

Além do fluxo principal, existem comandos úteis para depuração e testes rápidos:

```powershell
python -m src.main --inspect-datasets
python -m src.main --run-preprocessing
python -m src.main --test-mlp-classifier --dataset breast_cancer
python -m src.main --test-rna-models --dataset breast_cancer
python -m src.main --test-fuzzy-cmeans --dataset breast_cancer
python -m src.main --test-sugeno --dataset breast_cancer
```

Esses comandos não são necessários para reproduzir o experimento final, mas ajudam a validar partes isoladas do projeto.

## Observações finais

- O `README` serve como guia geral do projeto.
- O detalhamento metodológico completo, a justificativa das escolhas e a discussão crítica dos resultados devem ficar no artigo.
- O pipeline principal atual já está preparado para reprodução experimental com um único comando.
