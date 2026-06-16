#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/fnn/cliptta/ttavlm/data}"
SAVE_ROOT="${SAVE_ROOT:-/media/fnn/cliptta/ttavlm/result/officehome_osda}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
KNOWN_RATIO="${KNOWN_RATIO:-0.5}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-3}"

DOMAINS=("Art" "Clipart" "Product" "Real World")
mkdir -p "${SAVE_ROOT}"

for SOURCE_DOMAIN in "${DOMAINS[@]}"; do
  SOURCE_TAG="${SOURCE_DOMAIN// /_}"
  SOURCE_CKPT="${SAVE_ROOT}/source/${SOURCE_TAG}/clip_ViT-B_16_known${KNOWN_RATIO}_seed${SEED}.pt"

  if [[ ! -f "${SOURCE_CKPT}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.source_train \
      --dataset officehome \
      --source_domain "${SOURCE_DOMAIN}" \
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

  for TARGET_DOMAIN in "${DOMAINS[@]}"; do
    if [[ "${TARGET_DOMAIN}" == "${SOURCE_DOMAIN}" ]]; then
      continue
    fi

    TARGET_TAG="${TARGET_DOMAIN// /_}"
    RUN_SAVE_ROOT="${SAVE_ROOT}/${SOURCE_TAG}_to_${TARGET_TAG}"
    mkdir -p "${RUN_SAVE_ROOT}"

    CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.main \
      --exp_name "officehome_${SOURCE_TAG}_to_${TARGET_TAG}_osda_cliptta" \
      --dataroot "${DATA_ROOT}" \
      --save_root "${RUN_SAVE_ROOT}" \
      --source_checkpoint "${SOURCE_CKPT}" \
      --dataset officehome \
      --shift_type "${TARGET_DOMAIN}" \
      --adaptation cliptta \
      --base_model_name clip-ViT-B/16 \
      --source_free_open_set \
      --known_class_ratio "${KNOWN_RATIO}" \
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
      --seeds "${SEED}"
  done
done
