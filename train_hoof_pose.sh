#!/usr/bin/env bash
set -e  # stop immediately on error

echo "======================================"
echo "🐎 Horse Hoof RTMPose Training Pipeline"
echo "======================================"

# Activate env (optional but recommended)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_mmpose_env

# Paths
ANN_DIR="data/annotations"
TRAIN_JSON="$ANN_DIR/train.json"
VAL_JSON="$ANN_DIR/val.json"
CFG="mmpose/custom_configs/rtmpose_hoof_4kp.py"

echo "🔍 Validating COCO annotations..."

python normalize_coco_points.py "$TRAIN_JSON"
python normalize_coco_points.py "$VAL_JSON"

echo "✅ COCO annotations validated"

echo "🚀 Starting training..."
python mmpose/tools/train.py "$CFG"

echo "🎉 Training completed successfully"
