#!/bin/bash
# idr0111: train + test (leave-one-day-out) on the collected features. CPU only, ~1-2 h per series.
#   cd SI_Project/idr0111
#   sbatch run_train.sh mutants     # main test: MCF10A vs PTEN-/-, KRas, KRas+PTEN-/-
#   sbatch run_train.sh mda231      # second test: MCF10A vs MDA-MB-231
# result: results/<series>/report.md
#SBATCH --job-name=idr0111_train
#SBATCH --account=sutravek
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
PROJECT=${PROJECT:-$HOME/SI_Project}
VENV=${VENV:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/.venv_cpsam}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs results
module load python/3.12.3 >/dev/null 2>&1 || true
export PATH="$VENV/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
python -c "import shap" 2>/dev/null || pip install -q shap
S=${1:?usage: sbatch run_train.sh mutants|mda231}
python idr111_train.py --features features --series "$S" --out "results/$S" ${TRAIN_ARGS:-}
