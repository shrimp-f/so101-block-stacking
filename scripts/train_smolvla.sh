#!/usr/bin/env bash
# SmolVLA (lerobot/smolvla_base) を 2 段積みのデモデータセットで追加学習する。
# データセットのカメラ front / wrist を、smolvla_base の camera1 / camera2 に対応させ、3 台目は空のカメラにする。
# 使い方 (コンテナ内): bash scripts/train_smolvla.sh [steps]
set -euo pipefail

STEPS=${1:-2000}
OUT=outputs/train/smolvla_stack2

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/so101_jenga_stack2 \
    --dataset.root=data/so101_jenga_stack2 \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --policy.empty_cameras=1 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --steps="$STEPS" \
    --log_freq=50 \
    --save_freq=500 \
    --num_workers=4 \
    --output_dir="$OUT" \
    --job_name=smolvla_stack2 \
    --wandb.enable=false
