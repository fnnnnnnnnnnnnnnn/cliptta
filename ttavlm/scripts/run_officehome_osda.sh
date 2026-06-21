#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/fnn/cliptta/ttavlm/data}"
SAVE_ROOT="${SAVE_ROOT:-/media/fnn/cliptta/ttavlm/result/officehome_vit_osda}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
KNOWN_RATIO="${KNOWN_RATIO:-0.5}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-10}"
K_UNKNOWN="${K_UNKNOWN:-}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-clip-ViT-B/16}"

DOMAINS=("Art" "Clipart" "Product" "Real World")
mkdir -p "${SAVE_ROOT}"

for SOURCE_DOMAIN in "${DOMAINS[@]}"; do
  SOURCE_TAG="${SOURCE_DOMAIN// /_}"
  SOURCE_CKPT="${SAVE_ROOT}/source/${SOURCE_TAG}/clip_vit_b16_known${KNOWN_RATIO}_seed${SEED}.pt"
  if [[ ! -f "${SOURCE_CKPT}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.fused_osda source-train \
      --dataset officehome \
      --source_domain "${SOURCE_DOMAIN}" \
      --base_model_name "${BASE_MODEL_NAME}" \
      --dataroot "${DATA_ROOT}" \
      --save_root "${SAVE_ROOT}" \
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

    EXTRA_ARGS=()
    if [[ -n "${K_UNKNOWN}" ]]; then
      EXTRA_ARGS+=(--k_unknown "${K_UNKNOWN}")
    fi

    CUDA_VISIBLE_DEVICES="${GPU}" python -m ttavlm.fused_osda adapt \
      --dataset officehome \
      --source_domain "${SOURCE_DOMAIN}" \
      --target_domain "${TARGET_DOMAIN}" \
      --dataroot "${DATA_ROOT}" \
      --save_root "${SAVE_ROOT}" \
      --known_class_ratio "${KNOWN_RATIO}" \
      --batch_size 64 \
      --ood_batch_size 64 \
      --workers 4 \
      --seed "${SEED}" \
      --source_checkpoint "${SOURCE_CKPT}" \
      --base_model_name "${BASE_MODEL_NAME}" \
      "${EXTRA_ARGS[@]}"
  done
done
