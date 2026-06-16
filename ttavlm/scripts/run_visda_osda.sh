#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/fnn/cliptta/ttavlm/data}"
SAVE_ROOT="${SAVE_ROOT:-/media/fnn/cliptta/ttavlm/result/visda_osda}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
KNOWN_RATIO="${KNOWN_RATIO:-0.5}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-3}"

SOURCE_CKPT="${SAVE_ROOT}/source/train/clip_ViT-B_16_known${KNOWN_RATIO}_seed${SEED}.pt"
mkdir -p "${SAVE_ROOT}"

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.source_train \
    --dataset visda \
    --source_domain train \
    --dataroot "${DATA_ROOT}" \
    --save_root "${SAVE_ROOT}" \
    --base_model_name clip-ViT-B/16 \
    --known_class_ratio "${KNOWN_RATIO}" \
    --epochs "${SOURCE_EPOCHS}" \
    --batch_size 64 \
    --workers 4 \
    --lr 1e-4 \
    --seed "${SEED}" \
    --output "${SOURCE_CKPT}"
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.main \
  --exp_name visda_train_to_validation_osda_cliptta \
  --dataroot "${DATA_ROOT}" \
  --save_root "${SAVE_ROOT}/train_to_validation" \
  --source_checkpoint "${SOURCE_CKPT}" \
  --dataset visda \
  --shift_type validation \
  --adaptation cliptta \
  --base_model_name clip-ViT-B/16 \
  --source_free_open_set \
  --known_class_ratio "${KNOWN_RATIO}" \
  --k_unknown 4 \
  --batch_size 64 \
  --ood_batch_size 64 \
  --workers 4 \
  --steps 1 \
  --optimizer_type adam \
  --lr 1e-3 \
  --beta_ood 0.1 \
  --beta_cluster 0.1 \
  --beta_nl 0.1 \
  --queue_size 8192 \
  --n_neighbors 3 \
  --seeds "${SEED}"
