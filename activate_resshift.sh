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
