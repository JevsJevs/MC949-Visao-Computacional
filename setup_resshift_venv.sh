#!/bin/bash

# Script para configurar ambiente virtual isolado para ResShift
# Uso: bash setup_resshift_venv.sh

set -e  # Parar em caso de erro

echo "================================================"
echo "Configurando ambiente virtual para ResShift"
echo "================================================"

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Diretório do projeto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ============================================================
# VERIFICAÇÃO DE PYTHON 3.10
# ResShift requer PyTorch 2.1.1, que só é compatível com Python 3.10
# ============================================================

echo -e "\n${GREEN}Verificando Python 3.10...${NC}"

# Detectar Python 3.10
PYTHON_CMD=""
PYTHON_VERSION=""
for cmd in python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | awk '{print $2}')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" = "3" ] && [ "$minor" = "10" ]; then
            PYTHON_CMD=$cmd
            PYTHON_VERSION=$version
            echo -e "${GREEN}✓ Python $version encontrado: $cmd${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}✗ Erro: Python 3.10 não encontrado!${NC}"
    echo -e ""
    echo -e "${YELLOW}Por que Python 3.10 é necessário?${NC}"
    echo -e "  • ResShift requer PyTorch 2.1.1"
    echo -e "  • PyTorch 2.1.1 só é compatível com Python 3.10"
    echo -e "  • Versões mais novas de Python (3.11+) não são suportadas"
    echo -e ""
    echo -e "${YELLOW}Como instalar Python 3.10:${NC}"
    echo -e "  Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo -e "  Fedora: sudo dnf install python3.10"
    echo -e "  MacOS: brew install python@3.10"
    echo -e ""
    exit 1
fi

# Nome do ambiente virtual com versão do Python
# Formato: resshift_env_py3.10
VENV_NAME="resshift_env_py$(echo $PYTHON_VERSION | cut -d. -f1,2)"

echo -e "${GREEN}✓ Ambiente virtual será: $VENV_NAME${NC}"

# Verificar se já existe
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}⚠ Ambiente virtual '$VENV_NAME' já existe.${NC}"
    read -p "Deseja removê-lo e criar um novo? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo -e "${YELLOW}Removendo ambiente existente...${NC}"
        rm -rf "$VENV_NAME"
    else
        echo -e "${RED}✗ Cancelado.${NC}"
        exit 1
    fi
fi

# Criar ambiente virtual
echo -e "\n${GREEN}1. Criando ambiente virtual com $PYTHON_CMD...${NC}"
echo -e "${YELLOW}   Usando Python $PYTHON_VERSION (necessário para PyTorch 2.1.1)${NC}"
$PYTHON_CMD -m venv "$VENV_NAME"

# Ativar ambiente virtual
echo -e "${GREEN}2. Ativando ambiente virtual...${NC}"
source "$VENV_NAME/bin/activate"

# Atualizar pip
echo -e "\n${GREEN}3. Atualizando pip...${NC}"
pip install --upgrade pip

# Instalar PyTorch 2.1.1 (versão compatível com ResShift)
echo -e "\n${GREEN}4. Instalando PyTorch 2.1.1...${NC}"
# Detectar CUDA disponível
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}   CUDA detectado, instalando versão GPU...${NC}"
    pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
elif python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}   CUDA disponível, instalando versão GPU...${NC}"
    pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}   CUDA não detectado, instalando versão CPU...${NC}"
    # Tentar primeiro o índice cu118 (tem mais versões disponíveis)
    pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118 || \
    # Se falhar, tentar sem especificar índice (usa PyPI)
    pip install torch==2.1.1 torchvision==0.16.1
fi

# Criar diretório external se não existir
echo -e "\n${GREEN}5. Preparando diretório para ResShift...${NC}"
mkdir -p external

# Clonar ResShift se ainda não existe
if [ ! -d "external/ResShift" ]; then
    echo -e "${GREEN}6. Clonando repositório ResShift...${NC}"
    cd external
    git clone https://github.com/zsyOAOA/ResShift.git
    cd ..
else
    echo -e "${YELLOW}⚠ ResShift já existe em external/ResShift${NC}"
fi

# Instalar dependências do ResShift
echo -e "\n${GREEN}7. Instalando dependências do ResShift...${NC}"
pip install numpy==1.24.3  # Compatível com PyTorch 2.1.1 e scipy 1.9.3
pip install scipy==1.9.3
pip install opencv-python matplotlib Pillow
pip install timm pandas scikit-learn scikit-image
pip install lpips loguru omegaconf einops imageio albumentations

# Instalar xformers (pode falhar sem CUDA, mas não é crítico)
echo -e "\n${GREEN}8. Tentando instalar xformers (opcional)...${NC}"
pip install xformers==0.0.23 || echo -e "${YELLOW}⚠ xformers não instalado (requer CUDA). Não é crítico para CPU.${NC}"

# Criar script de ativação rápida
echo -e "\n${GREEN}9. Criando script de ativação...${NC}"
cat > activate_resshift.sh << EOF
#!/bin/bash
# Script para ativar ambiente virtual do ResShift
# Uso: source activate_resshift.sh
# 
# IMPORTANTE: Este ambiente usa Python $PYTHON_VERSION
# ResShift requer PyTorch 2.1.1, compatível apenas com Python 3.10

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

# Tentar encontrar o ambiente virtual correto
if [ -d "\$SCRIPT_DIR/$VENV_NAME" ]; then
    source "\$SCRIPT_DIR/$VENV_NAME/bin/activate"
elif [ -d "\$SCRIPT_DIR/resshift_env" ]; then
    # Fallback para nome antigo
    source "\$SCRIPT_DIR/resshift_env/bin/activate"
else
    echo "Erro: Ambiente virtual ResShift não encontrado!"
    echo "Execute: bash setup_resshift_venv.sh"
    return 1
fi

echo "✓ Ambiente ResShift ativado!"
echo "  Python: \$(python --version)"
echo "  PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Para desativar: deactivate"
EOF

chmod +x activate_resshift.sh

# Criar README para o venv
cat > $VENV_NAME/README.md << EOF
# Ambiente Virtual ResShift

Este é um ambiente Python isolado para executar o ResShift sem conflitos com o projeto principal.

**Python**: $PYTHON_VERSION  
**PyTorch**: 2.1.1 (compatível apenas com Python 3.10)

## Por que Python 3.10?

ResShift requer PyTorch 2.1.1, que só é compatível com Python 3.10.x.  
Versões mais novas (3.11+) não são suportadas por esta versão do PyTorch.

## Como Usar

### Ativar ambiente
\`\`\`bash
source activate_resshift.sh
# ou
source $VENV_NAME/bin/activate
\`\`\`

### Testar ResShift
\`\`\`bash
python src/canon/T4/process/test_one_model.py resshift
\`\`\`

### Desativar ambiente
\`\`\`bash
deactivate
\`\`\`

## Dependências Instaladas

- Python $PYTHON_VERSION
- PyTorch 2.1.1
- torchvision 0.16.1
- numpy 1.24.3
- scipy 1.9.3
- opencv-python, matplotlib, Pillow
- timm, pandas, scikit-learn, scikit-image
- lpips, loguru, omegaconf, einops, imageio, albumentations

## Observações

- Este ambiente é **isolado** do ambiente principal
- Use PyTorch 2.1.1 (vs 2.9.1 do projeto principal)
- Não interfere com outros modelos (Stable Diffusion, Kandinsky, etc.)
- Requer ~2GB de espaço em disco
EOF

# Verificar instalação
echo -e "\n${GREEN}10. Verificando instalação...${NC}"
python -c "import torch; print(f' PyTorch {torch.__version__} instalado')"
python -c "import torchvision; print(f' torchvision {torchvision.__version__} instalada')"
python -c "import numpy; print(f' NumPy {numpy.__version__} instalado')"
python -c "import cv2; print(f' OpenCV {cv2.__version__} instalado')"

# Desativar ambiente
deactivate

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN} Configuração completa!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "Para usar o ResShift:"
echo -e "  ${YELLOW}1. Ativar ambiente:${NC}     source activate_resshift.sh"
echo -e "  ${YELLOW}3. Desativar ambiente:${NC}  deactivate"
echo ""
echo -e "Ambiente instalado em: ${YELLOW}$PROJECT_DIR/$VENV_NAME${NC}"
echo -e "Tamanho aproximado: ~2GB"
echo ""
