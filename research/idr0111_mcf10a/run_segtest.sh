#!/bin/bash
# idr0111 segmentation test on 3 videos (GPU, ~10 min).   cd SI_Project/idr0111 && sbatch run_segtest.sh
#SBATCH --job-name=idr0111_seg
#SBATCH --account=sutravek
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
PROJECT=${PROJECT:-$HOME/SI_Project}
VENV=${VENV:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/.venv_cpsam}
export CELLPOSE_LOCAL_MODELS_PATH=${CELLPOSE_LOCAL_MODELS_PATH:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/cellpose_models}
FT=${FT_MODEL:-$PROJECT/CTC_Data/Thesis/2D+3D/models/cpsam/models/cpsam_ft_joint}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
module load python/3.12.3 >/dev/null 2>&1 || true
module load cuda/12.9 >/dev/null 2>&1 || true
export PATH="$VENV/bin:$PATH"
python -c "import tifffile" 2>/dev/null || pip install -q tifffile
python segtest_idr0111.py --data data --out segtest --ft-model "$FT"
