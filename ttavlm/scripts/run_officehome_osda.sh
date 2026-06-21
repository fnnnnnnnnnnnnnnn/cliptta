#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/fnn/cliptta/ttavlm/data}"
SAVE_ROOT="${SAVE_ROOT:-/media/fnn/cliptta/ttavlm/result/officehome_vit_osda}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
KNOWN_RATIO="${KNOWN_RATIO:-0.5}"
NUM_PRIVATE_PROTOTYPES="${NUM_PRIVATE_PROTOTYPES:-}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/16}"

DOMAINS=("Art" "Clipart" "Product" "Real World")
mkdir -p "${SAVE_ROOT}"

for SOURCE_DOMAIN in "${DOMAINS[@]}"; do
  for TARGET_DOMAIN in "${DOMAINS[@]}"; do
    if [[ "${TARGET_DOMAIN}" == "${SOURCE_DOMAIN}" ]]; then
      continue
    fi

    EXTRA_ARGS=()
    if [[ -n "${NUM_PRIVATE_PROTOTYPES}" ]]; then
      EXTRA_ARGS+=(--num_private_prototypes "${NUM_PRIVATE_PROTOTYPES}")
    fi

    CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.vit_osda adapt-eval \
      --dataset officehome \
      --source_domain "${SOURCE_DOMAIN}" \
      --target_domain "${TARGET_DOMAIN}" \
      --dataroot "${DATA_ROOT}" \
      --save_root "${SAVE_ROOT}" \
      --known_class_ratio "${KNOWN_RATIO}" \
      --batch_size 64 \
      --workers 4 \
      --seed "${SEED}" \
      --clip_model "${CLIP_MODEL}" \
      "${EXTRA_ARGS[@]}"
  done
done
