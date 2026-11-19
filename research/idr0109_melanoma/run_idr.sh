#!/bin/bash
# idr0109 (melanoma metastatic efficiency): check + unpack, per-cell features, ViT features, leave-one-tumour-out.
#
#   cd $PROJECT/Metastasis
#   bash run_idr.sh submit       # EASIEST: queues all four steps in order (run on the login node, not with sbatch)
#
# or one step at a time:
#   sbatch run_idr.sh unpack     # check the downloads, unpack images + masks       (CPU, a few hours)
#   sbatch run_idr.sh build      # hand-made features per tracked cell              (CPU, ~1 h)
#   sbatch --partition=gpu --gres=gpu:nvidia_h100_nvl:1 run_idr.sh embed
#                                # Cellpose-SAM ViT features per tracked cell       (GPU, ~1 h)
#   sbatch run_idr.sh train      # compare hand / ViT+movement / hybrid, reports    (CPU, ~30 min)
#   sbatch run_idr.sh normal     # melanoma (abnormal) vs normal melanocytes        (CPU, ~1 h)
#
#SBATCH --job-name=idr0109
#SBATCH --account=sutravek
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

PROJECT=${PROJECT:-$HOME/SI_Project}
DATA=${DATA:-$PROJECT/datasets/idr0109}
UNP=${UNP:-$DATA/unpacked}
OUT=${OUT:-$PROJECT/Metastasis/out}
VENV=${VENV:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/.venv_cpsam}
export CELLSTATE_DIR=${CELLSTATE_DIR:-$PROJECT/CellState}                              # cellstate_build.py (encoder)
export CELL_MODEL=${CELL_MODEL:-$PROJECT/CTC_Data/Thesis/models/cpsam/models/cpsam_ft_joint}  # website whole-cell model
export CELLPOSE_LOCAL_MODELS_PATH=${CELLPOSE_LOCAL_MODELS_PATH:-$PROJECT/CTC_Data/Models/Cellpose-SAM/ZeroShot/cellpose_models}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
CPUS=${SLURM_CPUS_PER_TASK:-8}
log() { echo "[$(date '+%F %T')] $*"; }

module load python/3.12.3 >/dev/null 2>&1 || true
export PATH="$VENV/bin:$PATH"
mkdir -p "$CODE_DIR/logs" "$OUT"

check() {   # compare the downloads with SHASUMS (sha1 or sha256, chosen by hash length)
  cd "$DATA"
  [ -f 20210408-ftp/SHASUMS ] || { log "no SHASUMS file, skipping the check"; return; }
  while read -r h f; do
    [ -z "${h:-}" ] && continue
    loc=$(find . -name "$(basename "$f")" -not -path "./unpacked/*" | head -1)
    [ -z "$loc" ] && { log "not downloaded (fine if not needed): $f"; continue; }
    case ${#h} in 40) c=$(sha1sum "$loc" | cut -d' ' -f1);; 64) c=$(sha256sum "$loc" | cut -d' ' -f1);; *) c="?";; esac
    if [ "$c" = "$h" ]; then log "OK   $loc"; else log "BAD  $loc (expected $h, got $c) -> delete it and download again"; exit 1; fi
  done < 20210408-ftp/SHASUMS
}

unpack() {
  check
  mkdir -p "$UNP/roi" "$UNP/mask"
  local BZ="bzip2"
  command -v lbzip2 >/dev/null && BZ="lbzip2 -n $CPUS"
  command -v pbzip2 >/dev/null && [ "$BZ" = "bzip2" ] && BZ="pbzip2 -p$CPUS"
  log "unpacking single-cell images with: $BZ"
  tar --use-compress-program="$BZ" -xf "$DATA/20210408-ftp/ROI/imageROIs.tar.bz2" -C "$UNP/roi"
  log "unpacking masks"
  tar -xzf "$DATA/20210421/masks.tgz" -C "$UNP/mask"
  log "files: images $(find "$UNP/roi" -type f | wc -l), masks $(find "$UNP/mask" -type f | wc -l)"
  du -sh "$UNP"/*
}

build() {
  log "features per tracked cell"
  python "$CODE_DIR/idr_build.py" \
    --roi-table "$DATA/20210408-ftp/ROI/ImageFileTable_ROI.txt" \
    --mask-table "$DATA/20210416-ftp/ImageFileTable_roiMask_15APR2021.txt" \
    --roi-root "$UNP/roi" --mask-root "$UNP/mask" --out "$OUT/features" --workers "$CPUS" ${BUILD_ARGS:-}
}

embed() {
  [ -f "$CELLSTATE_DIR/cellstate_build.py" ] || { log "cellstate_build.py not in $CELLSTATE_DIR (set CELLSTATE_DIR=...)"; exit 1; }
  [ -f "$CELL_MODEL" ] || { log "whole-cell model not found: $CELL_MODEL (set CELL_MODEL=...)"; exit 1; }
  module load cuda/12.9 >/dev/null 2>&1 || true
  log "ViT features per tracked cell (model: $CELL_MODEL)"
  python "$CODE_DIR/idr_embed.py" \
    --roi-table "$DATA/20210408-ftp/ROI/ImageFileTable_ROI.txt" \
    --mask-table "$DATA/20210416-ftp/ImageFileTable_roiMask_15APR2021.txt" \
    --roi-root "$UNP/roi" --mask-root "$UNP/mask" --out "$OUT/embed" \
    --model "$CELL_MODEL" --cellstate-dir "$CELLSTATE_DIR" ${EMBED_ARGS:-}
}

train() {
  local E=""
  [ -f "$OUT/embed/embeddings.npz" ] && E="--embed $OUT/embed/embeddings.npz" || log "no ViT features yet: training the hand-made model only"
  log "leave-one-tumour-out training and model comparison"
  local TO=${TRAIN_OUT:-$OUT/train}
  python "$CODE_DIR/idr_train.py" --tracks "$OUT/features/tracks.csv.gz" $E --out "$TO" ${TRAIN_ARGS:-}
  log "comparison: $TO/comparison.md   per model: $TO/<model>/report.md"
}

normal() {   # melanoma (abnormal) vs normal melanocytes, leave-one-family-out; NORMAL_ARGS="--check-only" = counts only
  local NO=${NORMAL_OUT:-$OUT/normal}
  log "normal vs abnormal: melanoma cells vs melanocytes"
  python "$CODE_DIR/idr_normal.py" --tracks "$OUT/features/tracks.csv.gz" --embed "$OUT/embed/embeddings.npz" --out "$NO" ${NORMAL_ARGS:-}
  log "result: $NO/comparison.md"
}

submit() {   # login node: queue unpack -> (build on CPU, embed on GPU) -> train
  cd "$CODE_DIR"
  local j1 j2 j3 j4
  j1=$(sbatch --parsable run_idr.sh unpack)
  j2=$(sbatch --parsable --dependency=afterok:$j1 run_idr.sh build)
  j3=$(sbatch --parsable --dependency=afterok:$j1 --partition=gpu --gres=gpu:nvidia_h100_nvl:1 --cpus-per-task=8 run_idr.sh embed)
  j4=$(sbatch --parsable --dependency=afterok:$j2:$j3 run_idr.sh train)
  echo "queued: unpack $j1 -> build $j2 + embed $j3 (GPU) -> train $j4"
  echo "watch:  squeue -u ${USER:-$(whoami)}      logs: $CODE_DIR/logs/idr0109_<job>.out"
}

case "${1:-all}" in
  check) check ;;
  unpack) unpack ;;
  build) build ;;
  embed) embed ;;
  train) train ;;
  normal) normal ;;
  submit) submit; exit 0 ;;
  all) unpack; build; train ;;
  *) echo "usage: bash run_idr.sh submit   |   sbatch run_idr.sh {check|unpack|build|embed|train|normal|all}"; exit 2 ;;
esac
log "done: ${1:-all}"