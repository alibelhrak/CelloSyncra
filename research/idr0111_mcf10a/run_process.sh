#!/bin/bash
# idr0111: segment + track + measure + encoder features for all 198 videos, one GPU job per video (~5 min each).
#   cd SI_Project/idr0111
#   sbatch run_process.sh                 # all 198 videos, at most 10 at a time
#   sbatch --array=0-1 run_process.sh     # quick test on 2 videos first
# finished videos are skipped if you submit again. Then: python idr111_collect.py
#SBATCH --job-name=idr0111_cells
#SBATCH --account=sutravek
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-197%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
PROJECT=${PROJECT:-$HOME/SI_Project}
VENV=${VENV:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/.venv_cpsam}
export CELLPOSE_LOCAL_MODELS_PATH=${CELLPOSE_LOCAL_MODELS_PATH:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/cellpose_models}
export CELLSTATE_DIR=${CELLSTATE_DIR:-$PROJECT/CellState}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs cells
module load python/3.12.3 >/dev/null 2>&1 || true
module load cuda/12.9 >/dev/null 2>&1 || true
export PATH="$VENV/bin:$PATH"
[ -f "$CELLSTATE_DIR/cellstate_build.py" ] || { echo "cellstate_build.py not in $CELLSTATE_DIR (set CELLSTATE_DIR=...)"; exit 1; }
python idr111_process.py --index "${SLURM_ARRAY_TASK_ID:-0}" --data data --out cells --cellstate-dir "$CELLSTATE_DIR" ${PROCESS_ARGS:-}
