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

**Modelos Principais (Core):**
- Stable Diffusion Inpainting
- Paint-by-Example
- Kandinsky 2.2 Inpainting

**Modelos Opcionais:**
- ResShift (requer instalação adicional)

### Testando os Modelos

```bash
# Teste rápido (recomendado)
python src/canon/T4/process/test_models.py --quick

# Teste apenas modelos principais
python src/canon/T4/process/test_models.py --core-only

# Teste completo com GPU
python src/canon/T4/process/test_models.py cuda
```

### Configurando ResShift (Opcional)

O ResShift requer um ambiente virtual isolado devido a incompatibilidade de versão do PyTorch.

```bash
# Instalação automática (recomendado)
bash setup_resshift_venv.sh

# Ativar o ambiente quando necessário
source activate_resshift.sh

# Testar
python src/canon/T4/process/test_one_model.py resshift

# Desativar ambiente
deactivate
```

### Uso Básico

```python
from canon.T4 import get_model
from PIL import Image

image = Image.open("data/T4/imagens/antiga_1.jpg")
mask = Image.open("data/T4/mascaras/antiga_1_mask_1.png")

model = get_model("stable_diffusion", device="cuda")
result = model.inpaint(image, mask)
result["image"].save("data/T4/results/resultado.png")
```

### Documentação Completa

- `docs/T4_Models_Documentation.md` - Documentação técnica
- `docs/T4_Integration_Guide.md` - Guia de integração
- `docs/T4_Testing_Guide.md` - Guia de testes
- `docs/T4_Implementation_Status.md` - Status da implementação