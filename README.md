# MC949-Visao-Computacional

Repositório com os projetos da disciplina MC949/MO4446 - Visão Computacional.

## Estrutura de Diretórios

Template de projeto inspirado no [Cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/#directory-structure).

- `data/`
    - `raw/`: Dataset original em sua forma inalterada
    - `interim/`: Versões pré-processadas dos dados originais
    - `results/`: Resultados finais do projeto
- `docs/`: Documentação do projeto
- `models/`: Modelos que serão criados
- `notebooks/`: Notebooks de playground para fins de pesquisa
- `src/`: Código definitivo do projeto

Como o repositório armazena o código referente a 4 projetos distintos, cada um desses diretórios foi dividido em T1, T2, T3 e T4. Com isso, a estrutura do repositório é a seguinte:

```txt
├── data
│   ├── T1
│   |   ├── interim
│   |   ├── raw
│   |   └── results
│   ├── T2
│   |   ├── interim
│   |   ├── raw
│   |   └── results
│   ├── T3
│   |   ├── interim
│   |   ├── raw
│   |   └── results
│   └── T4
|       ├── interim
│       ├── raw
│       └── results
├── docs
├── notebooks
│   ├── T1
│   ├── T2
│   ├── T3
│   └── T4
├── requirements.txt
├── run.sh
└── src
    ├── canon
    │   ├── T1
    │   ├── T2
    │   ├── config.py
    │   ├── download_data.py
    │   └── utils
    └── pyproject.toml
```

## Execução do Projeto T2

Para executar o projeto, foi disponibilizado um script `run.sh` na raiz do repositório. A execução do script realiza as seguintes etapas:

1. Criação do ambiente virtual e instalação das bibliotecas necessárias
2. Download dos dados
3. Execução da pipeline do projeto

Os seguintes comandos devem ser executados na raiz do repositório:

```bash
chmod +x run.sh
./run.sh
```
