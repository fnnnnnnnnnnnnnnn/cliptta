#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/fnn/cliptta/ttavlm/data}"
SAVE_ROOT="${SAVE_ROOT:-/media/fnn/cliptta/ttavlm/result/officehome_osda}"
GPU="${GPU:-0}"

mkdir -p "${SAVE_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.main \
  --exp_name officehome_osda_cliptta \
  --dataroot "${DATA_ROOT}" \
  --save_root "${SAVE_ROOT}" \
  --dataset officehome \
  --shift_type Art Clipart Product "Real World" \
  --adaptation cliptta \
  --base_model_name clip-ViT-B/16 \
  --source_free_open_set \
  --known_class_ratio 0.5 \
  --k_unknown 8 \
  --batch_size 64 \
  --ood_batch_size 64 \
  --workers 4 \
  --steps 1 \
  --optimizer_type adam \
  --lr 1e-3 \
  --beta_ood 0.1 \
  --beta_cluster 0.1 \
  --beta_nl 0.1 \
  --queue_size 16384 \
  --n_neighbors 3 \
  --seeds 42
