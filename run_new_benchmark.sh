#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
cd mmpose
CHECKPOINT="work_dirs/rtmpose_hoof_new_only_experiment/epoch_300.pth"
CONFIG="custom_configs/rtmpose_hoof_new_only.py"

echo "--- RUNNING NEW MODEL BENCHMARK ---"
for i in {1..6}
do
    IMG="/home/chetan/Desktop/demo$i.png"
    if [ -f "$IMG" ]; then
        echo "Processing demo$i.png..."
        /home/chetan/miniconda3/envs/env_mmpose_env/bin/python demo/inference_on_new_image_refined.py "$IMG" --config "$CONFIG" --checkpoint "$CHECKPOINT" | grep -E "✅|Zone"
    else
        echo "demo$i.png not found."
    fi
done
