# MC949-Visao-Computacional

Repositório com os projetos da disciplina MC949/MO4446 - Visão Computacional.

## Estrutura de Diretórios

Template de projeto inspirado no [Cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/#directory-structure).

- `data/`
    - `raw/`: Dataset original em sua forma inalterada
    - `interim/`: Versões pré-processadas dos dados originais
    - `results/`: Resultados finais do projeto
- `docs/`: Documentação do projeto
- `models/`: Modelos pré-treinados e checkpoints
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
|       ├── imagens
│       ├── mascaras
│       └── results
├── docs
├── models
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
    │   ├── T4
    │   │   ├── config
    │   │   ├── process (modelos de inpainting)
    │   │   └── utils.py
    │   ├── config.py
    │   ├── download_data.py
    │   └── utils
    └── pyproject.toml
```

## Execução dos Projetos

Para executar os projetos, foi disponibilizado um script `run.sh` na raiz do repositório. A execução do script realiza as seguintes etapas:

1. Criação do ambiente virtual e instalação das bibliotecas necessárias
2. Download dos dados do projeto especificado
3. Execução da pipeline (no caso do T2)

### Como Usar

Edite a variável `PROJECT` no arquivo `run.sh` (linha 5) para o projeto desejado (`T1`, `T2` ou `T4`), e execute os seguintes comandos na raiz do repositório:

```bash
chmod +x run.sh
./run.sh
```

**Exemplos:**
- Para T1: `PROJECT="T1"` - Baixa os dados e prepara o ambiente
- Para T2: `PROJECT="T2"` - Baixa os dados e executa automaticamente a pipeline de reconstrução 3D  
- Para T4: `PROJECT="T4"` - Baixa imagens e máscaras do Kaggle

## Projeto T4: Modelos de Difusão para Restauração de Imagens

O projeto T4 implementa modelos de difusão para tarefas de restauração e expansão de imagens (inpainting).

### Modelos Implementados

O projeto implementa 3 modelos de inpainting:

- **Stable Diffusion Inpainting**: Versátil e rápido, com suporte a prompts de texto opcionais
- **Paint-by-Example**: Restauração guiada por exemplos visuais
- **Kandinsky 2.2 Inpainting**: Otimizado para restauração de fotos vintage, com detecção automática de danos

### Restauração de Fotos Antigas

O projeto inclui utilitários especializados para restauração de fotos antigas:

- **Detecção Automática de Danos**: Identifica rachaduras, manchas e áreas deterioradas
- **Pré-processamento**: Redução de ruído e ajuste de contraste

Para mais detalhes, consulte `src/canon/T4/utils.py` (função `photo_restoration_utils`).