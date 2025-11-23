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

# Nome do ambiente virtual
VENV_NAME="resshift_env"

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
echo -e "\n${GREEN}1. Criando ambiente virtual...${NC}"
python3 -m venv "$VENV_NAME"

# Ativar ambiente virtual
echo -e "${GREEN}2. Ativando ambiente virtual...${NC}"
source "$VENV_NAME/bin/activate"

# Atualizar pip
echo -e "\n${GREEN}3. Atualizando pip...${NC}"
pip install --upgrade pip

# Instalar PyTorch 2.1.1 (versão compatível com ResShift)
echo -e "\n${GREEN}4. Instalando PyTorch 2.1.1...${NC}"
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu

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
pip install scipy==1.9.3
pip install numpy opencv-python matplotlib Pillow
pip install timm pandas scikit-learn scikit-image
pip install lpips loguru omegaconf einops imageio albumentations

# Instalar xformers (pode falhar sem CUDA, mas não é crítico)
echo -e "\n${GREEN}8. Tentando instalar xformers (opcional)...${NC}"
pip install xformers==0.0.23 || echo -e "${YELLOW}⚠ xformers não instalado (requer CUDA). Não é crítico para CPU.${NC}"

# Criar script de ativação rápida
echo -e "\n${GREEN}9. Criando script de ativação...${NC}"
cat > activate_resshift.sh << 'EOF'
#!/bin/bash
# Script para ativar ambiente virtual do ResShift
# Uso: source activate_resshift.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/resshift_env/bin/activate"

echo "✓ Ambiente ResShift ativado!"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Para testar: python src/canon/T4/process/test_one_model.py resshift"
echo "Para desativar: deactivate"
EOF

chmod +x activate_resshift.sh

# Criar README para o venv
cat > resshift_env/README.md << 'EOF'
# Ambiente Virtual ResShift

Este é um ambiente Python isolado para executar o ResShift sem conflitos com o projeto principal.

## Como Usar

### Ativar ambiente
```bash
source activate_resshift.sh
# ou
source resshift_env/bin/activate
```

### Testar ResShift
```bash
python src/canon/T4/process/test_one_model.py resshift
```

### Desativar ambiente
```bash
deactivate
```

## Dependências Instaladas

- Python 3.x
- PyTorch 2.1.1 (CPU)
- torchvision 0.16.1
- scipy 1.9.3
- numpy, opencv-python, matplotlib, Pillow
- timm, pandas, scikit-learn, scikit-image
- lpips, loguru, omegaconf, einops, imageio, albumentations

## Observações

- Este ambiente é **isolado** do ambiente principal
- Use PyTorch 2.1.1 (vs 2.9.1 do projeto principal)
- Não interfere com outros modelos (Stable Diffusion, ControlNet, etc.)
- Requer ~2GB de espaço em disco
EOF

# Verificar instalação
echo -e "\n${GREEN}10. Verificando instalação...${NC}"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} instalado')"
python -c "import torchvision; print(f'✓ torchvision {torchvision.__version__} instalada')"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__} instalado')"
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__} instalado')"

# Desativar ambiente
deactivate

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}✓ Configuração completa!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "Para usar o ResShift:"
echo -e "  ${YELLOW}1. Ativar ambiente:${NC}     source activate_resshift.sh"
echo -e "  ${YELLOW}2. Testar modelo:${NC}       python src/canon/T4/process/test_one_model.py resshift"
echo -e "  ${YELLOW}3. Desativar ambiente:${NC}  deactivate"
echo ""
echo -e "Ambiente instalado em: ${YELLOW}$PROJECT_DIR/$VENV_NAME${NC}"
echo -e "Tamanho aproximado: ~2GB"
echo ""
