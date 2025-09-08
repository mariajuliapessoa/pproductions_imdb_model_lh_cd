# LH_CD_MARIAJULIAPESSOA

## Descrição do Projeto

Este projeto contém a resolução do **Desafio Cientista de Dados** da Indicium, aplicado à base de dados cinematográfica da PProductions. O objetivo é realizar uma análise exploratória detalhada, responder perguntas estratégicas sobre filmes e desenvolver um modelo preditivo para a nota do IMDB.

O projeto segue a **estrutura Cookiecutter Data Science (CCDS v2)** para garantir modularidade, reprodutibilidade e boas práticas de engenharia de dados e ciência de dados.

---

## Estrutura do Projeto

```

lh\_cd\_mariajuliapessoa/
├── LICENSE
├── Makefile
├── README.md
├── data/
│   ├── interim/       # Dados transformados intermediários
│   ├── processed/     # Dados prontos para modelagem
│   └── raw/           # Base de dados original
├── docs/              # Documentação adicional
├── models/            # Modelos treinados e arquivo .pkl final
├── notebooks/         # Jupyter notebooks organizados por fase
├── pyproject.toml     # Configuração do projeto
├── references/        # Material de apoio, papers e manuais
├── reports/
│   └── figures/       # Gráficos gerados
├── requirements.txt   # Pacotes necessários
├── setup.cfg          # Configuração de linting/formatting
└── lh\_cd\_mariajuliapessoa/   # Código fonte reutilizável
├── **init**.py
├── config.py
├── dataset.py
├── features.py
├── modeling/
│   ├── **init**.py
│   ├── train.py
│   └── predict.py
└── plots.py

````

---

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seuusuario/lh_cd_mariajuliapessoa.git
cd lh_cd_mariajuliapessoa
````

2. Crie e ative o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Abra os notebooks para exploração e execução:

```bash
jupyter notebook notebooks/
```

---

## Conteúdo do Projeto

### 1. Exploração de Dados (EDA)

* Notebooks `0.*` contêm análise exploratória da base de dados.
* Objetivo: identificar padrões, distribuições e correlações.
* Inclusão de dados externos quando relevante.

### 2. Análise Estratégica

* Resposta às perguntas do desafio:

  * Filme recomendado sem conhecer a pessoa.
  * Principais fatores relacionados a alto faturamento.
  * Insights da coluna `Overview` e inferência de gênero.
  * Planejamento de previsão do IMDB Rating (tipo de problema, variáveis, modelo e métricas).

### 3. Modelagem

* Notebooks `3.*` e scripts `modeling/train.py` e `modeling/predict.py`.
* Desenvolvimento de modelo preditivo para nota do IMDB.
* Modelo salvo em `models/imdb_model.pkl`.

### 4. Relatórios e Visualizações

* Relatórios em PDF/HTML extraídos de notebooks.
* Gráficos prontos para apresentação em `reports/figures`.

### 5. Código Modular

* Todo código reutilizável está dentro do módulo `lh_cd_mariajuliapessoa/`.
* Facilita importação em notebooks e scripts:

```python
from lh_cd_mariajuliapessoa.dataset import load_data
from lh_cd_mariajuliapessoa.features import create_features
from lh_cd_mariajuliapessoa.plots import plot_distribution
```

---

## Execução

1. Para rodar notebooks: `jupyter notebook notebooks/`
2. Para treinar o modelo diretamente pelo script:

```bash
python -m lh_cd_mariajuliapessoa.modeling.train
```

3. Para gerar predições com o modelo salvo:

```bash
python -m lh_cd_mariajuliapessoa.modeling.predict
```

---

## Entregáveis

* `README.md` com instruções de uso.
* `requirements.txt` com todas as dependências.
* Notebooks com EDA e análises estatísticas.
* Scripts de modelagem (`train.py` e `predict.py`).
* Modelo final salvo como `.pkl` (`models/imdb_model.pkl`).
* Relatórios PDF/HTML com visualizações.

---

## Autor

Maria Julia Pessoa Cunha
Orientador: João Victor Tinoco de Souza Abreu

---

## Licença

Este projeto está sob licença MIT. Consulte o arquivo `LICENSE` para detalhes.

```

---

Se você quiser, eu posso **gerar também o Makefile inicial pronto**, já com comandos para criar ambiente, instalar dependências, rodar notebooks, treinar modelo e gerar gráficos — assim você só precisa colocar o código do desafio dentro da estrutura.  

Quer que eu faça isso?
```
