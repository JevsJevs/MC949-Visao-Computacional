#!/bin/bash
set -e

# ============================================================
# CONFIGURATION: Change this variable to select your project
# ============================================================
# Edit the line below to change which project to setup:
#   T1 - Panoramas
#   T2 - 3D Reconstruction
#   T4 - Inpainting Models
PROJECT="T4"

# Valid projects: T1, T2, T4
VALID_PROJECTS=("T1" "T2" "T4")

if [[ ! " ${VALID_PROJECTS[@]} " =~ " ${PROJECT} " ]]; then
  echo "[ERROR] Project '$PROJECT' is not supported."
  echo "Valid projects: ${VALID_PROJECTS[*]}"
  echo "Please edit the PROJECT variable in run.sh to one of: T1, T2, T4"
  exit 1
fi

echo "[INFO] Running setup for project: $PROJECT"

# Project-specific configurations
case "$PROJECT" in
  T1)
    DATA_DIR="data/T1"
    RUN_PIPELINE=false
    ;;
  T2)
    DATA_DIR="data/T2"
    IMAGE_DIR="data/T2/interim/GustavIIAdolf"
    RES_DIR="data/T2/interim/GustavIIAdolf/mainRun"
    RUN_PIPELINE=true
    ;;
  T4)
    DATA_DIR="data/T4"
    IMAGE_DIR="data/T4/imagens"
    MASK_DIR="data/T4/mascaras"
    RES_DIR="data/T4/results"
    RUN_PIPELINE=false
    ;;
esac

# ---------------------------
# Setup Python virtual environment
# ---------------------------
if [ ! -d ".venv" ]; then
  echo "[INFO] Creating Python virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
  else
    echo "[ERROR] requirements.txt not found. Exiting."
    exit 1
  fi
else
  echo "[INFO] Using existing virtual environment."
  source .venv/bin/activate
fi

# ---------------------------
# Patch dependencies (Fix basicsr vs torchvision)
# ---------------------------
PATCH_FILE=$(find .venv/lib/python3.*/site-packages/basicsr/data/ -name "degradations.py" 2>/dev/null | head -n 1)

if [ -n "$PATCH_FILE" ]; then
  echo "[INFO] Applying patch to basicsr: $PATCH_FILE"
  sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' "$PATCH_FILE"
else
  echo "[WARNING] basicsr/data/degradations.py not found. Skipping patch."
fi

# ---------------------------
# Download project data (if needed)
# ---------------------------
if [ ! -d "$DATA_DIR" ]; then
  echo "[INFO] Data folder '$DATA_DIR' not found, downloading project data..."
  python3 src/canon/download_data.py --project "$PROJECT"
else
  echo "[INFO] Data folder '$DATA_DIR' already exists, skipping download."
fi

# ---------------------------
# Run project-specific pipeline
# ---------------------------
if [ "$RUN_PIPELINE" = true ]; then
  case "$PROJECT" in
    T2)
      if [ ! -d "$IMAGE_DIR" ]; then
        echo "[ERROR] Dataset folder '$IMAGE_DIR' not found. Exiting."
        exit 1
      fi

      if [ ! -d "$RES_DIR" ]; then
        mkdir -p "$RES_DIR"
      fi

      echo "[INFO] Running 3D reconstruction pipeline on project 'T2', dataset 'GustavIIAdolf'..."
      python3 "src/canon/T2/main.py" \
        --image_dir "$IMAGE_DIR" \
        --res_dir "$RES_DIR" \
        --densify False

      echo "[INFO] Pipeline finished successfully! Results saved in ${RES_DIR}"
      ;;
  esac
else
  echo "[INFO] $PROJECT setup complete! Data available at: $DATA_DIR"
  echo ""
  
  case "$PROJECT" in
    T1)
      echo "[INFO] T1 data downloaded and ready to use."
      echo "[INFO] Check notebooks/T1/ for available notebooks."
      ;;
    T4)
      if [ ! -d "$RES_DIR" ]; then
        mkdir -p "$RES_DIR"
      fi
      
      echo "[INFO] Images folder: $IMAGE_DIR"
      echo "[INFO] Masks folder: $MASK_DIR"
      echo "[INFO] Results folder: $RES_DIR"
      echo ""
      echo "[INFO] To use the inpainting models, run:"
      echo "  jupyter notebook notebooks/T4/S2-1.0-hpbv-inpainting-models.ipynb"
      echo ""
      echo "[INFO] Or test models directly:"
      echo "  python3 src/canon/T4/process/test_models.py"
      ;;
  esac
fi
