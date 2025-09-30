#!/bin/bash
set -e  # exit on error

# # ---------------------------
# # Default values
# # ---------------------------
# PROJECT=""
# DATASET=""

# # ---------------------------
# # Parse arguments
# # ---------------------------
# while [[ $# -gt 0 ]]; do
#   key="$1"
#   case $key in
#     --project)
#       PROJECT="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     --dataset)
#       DATASET="$2"
#       shift # past argument
#       shift # past value
#       ;;
#     *)
#       echo "Unknown argument: $1"
#       echo "Usage: $0 --project <PROJECT_NAME> --dataset <DATASET_NAME>"
#       exit 1
#       ;;
#   esac
# done

# # ---------------------------
# # Validate arguments
# # ---------------------------
# if [ -z "$PROJECT" ] || [ -z "$DATASET" ]; then
#   echo "Error: both --project and --dataset must be provided"
#   echo "Usage: $0 --project <PROJECT_NAME> --dataset <DATASET_NAME>"
#   exit 1
# fi

# if [ "$PROJECT" != "T2" ]; then
#   echo "[ERROR] Project '$PROJECT' is not supported yet. Only 'T2' works for now."
#   exit 1
# fi

# # Build image directory path automatically
IMAGE_DIR="data/T2/interim/GustavIIAdolf"
RES_DIR="data/T2/interim/GustavIIAdolf/mainRun"

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
# Download project data (if needed)
# ---------------------------
if [ ! -d "data" ]; then
  echo "[INFO] Data folder not found, downloading project data..."
  python3 src/canon/download_data.py --project T2
else
  echo "[INFO] Data folder already exists, skipping download."
fi

# ---------------------------
# Check dataset folder exists
# ---------------------------
if [ ! -d "$IMAGE_DIR" ]; then
  echo "[ERROR] Dataset folder '$IMAGE_DIR' not found. Exiting."
  exit 1
fi

# ---------------------------
# Check output folder exists
# ---------------------------
if [ ! -d "$RES_DIR" ]; then
  mkdir -p "$RES_DIR"
fi

# ---------------------------
# Run pipeline
# ---------------------------
echo "[INFO] Running 3D reconstruction pipeline on project 'T2', dataset 'GustavIIAdolf'..."
python3 "src/canon/T2/main.py" \
  --image_dir "$IMAGE_DIR" \
  --res_dir "$RES_DIR" \
  --densify False

echo "[INFO] Pipeline finished successfully! Results saved in ${RES_DIR}"
