# Horse Hoof Keypoint Detection (MMPose)

This project trains a **custom MMPose RTMPose model** to detect **4 hoof keypoints**
from horse images using **COCO keypoint annotations**.

This README documents:
- ✅ The **working Linux + NVIDIA setup**
- ❌ Failed attempts (macOS, mmcv-lite, config mistakes)
- 🧠 Lessons learned so future developers don’t repeat the same mistakes

---

## 1. System Requirements (Verified)

| Component | Requirement |
|---------|------------|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA GPU (Tested on GTX 1660 SUPER – 6GB) |
| Driver | NVIDIA 535.xx |
| CUDA | 12.1 / 12.2 |
| Python | 3.10 |
| Conda | Miniconda |

> ⚠️ **macOS is NOT recommended** for MMPose training  
> See [Failed Attempts](#failed-attempts--lessons-learned)

---

## 2. GPU & CUDA Verification

```bash
nvidia-smi

Expected output:

GPU detected

Driver ≥ 535

CUDA ≥ 12.x

If nvidia-smi is missing:
sudo apt install nvidia-utils-535

Verify GPU visibility
lspci | grep -i nvidia

3. Conda Setup (Correct Way)
Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

Install loaction recommended
/home/<user>/miniconda3

Reload shell
source ~/.bashrc

Verify

conda --version

Create env:
conda create -n env_mmpose_env python=3.10 -y
conda activate env_mmpose_env

verify:
python --version

Insall Pytorch:
For Cuda 2.1

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

  python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF

Expected:
True
NVIDIA GeForce GTX 1660 SUPER

6. Install OpenMMLab Stack (CRITICAL)
MMEngine
pip install -U mmengine

MMCV (FULL, NOT LITE)
pip install -U mmcv \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html


Verify CUDA ops:

python - <<EOF
from mmcv.ops import MultiScaleDeformableAttention
print("MMCV CUDA ops OK")
EOF

7. Install MMPose
pip install -U mmpose


Verify:

python - <<EOF
import mmpose
print(mmpose.__version__)
EOF

8. Dataset Format

Annotation tool: CVAT

Format: COCO Keypoints

Keypoints: 4 hoof points

Dataset structure:

data/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── train/
    └── val/

9. Training
mim train mmpose configs/hoofs_keypoints/rtmpose_hoof_4kp.py


Checkpoints saved to:

work_dirs/rtmpose_hoof_4kp/

10. Adding More Data (Incremental Training)

When new images are available:

Annotate in CVAT

Export COCO JSON

Merge with existing annotations

Resume training

resume = True
load_from = "work_dirs/rtmpose_hoof_4kp/latest.pth"


🚫 Model does NOT need to retrain from scratch.

11. Minimum Dataset Guidance
Images	Outcome
~50	Overfitting risk (acceptable for POC)
150–300	Reasonable learning
500+	Stable generalization

With only 4 keypoints, fewer images are acceptable compared to full-body pose.

12. Understanding MMEngine Logs

Example:

(VERY_HIGH) RuntimeInfoHook
(LOW) ParamSchedulerHook


These are hook priorities, not data values.

Priority order:

VERY_HIGH > HIGH > NORMAL > LOW > VERY_LOW


They define execution order, not importance or performance.

13. Failed Attempts & Lessons Learned
❌ macOS (Intel / Apple Silicon)

mmcv CUDA ops unavailable

Frequent registry & build failures

No stable training pipeline

➡ Decision: macOS is unsuitable for MMPose training

❌ mmcv-lite

Error:

Fail to import MultiScaleDeformableAttention


Reason:

mmcv-lite lacks CUDA extensions

➡ Always install full mmcv

❌ Missing Registry Errors

Errors like:

RTMPoseBackbone not in registry
HeatmapDecoder not in KEYPOINT_CODECS


Cause:

Custom imports without registry registration

Incorrect config copy-paste

➡ Use official MMPose configs as base

❌ Conda Not Found in VS Code

Cause:

VS Code opened before conda initialization

Fix:

source ~/.bashrc


Restart VS Code terminal.

14. Final Recommendation

✔ Linux + NVIDIA GPU is the correct path

✔ Document failures as well as successes

❌ Do not attempt training on macOS

✔ Use this README as the single source of truth

Maintainers

Initial setup & debugging: Linux migration

Framework: MMPose + RTMPose

Annotation: CVAT

STEP 3 — Install MMEngine (core dependency)
pip install -U mmengine


Verify:

python - <<EOF
import mmengine
print(mmengine.__version__)
EOF

STEP 4 — Install MMCV (FULL version, CUDA enabled)

⚠️ This is where Mac failed. Linux will not.

For CUDA 12.1 + Torch 2.x:

pip install -U mmcv \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

Verify CUDA ops (VERY IMPORTANT)
python - <<EOF
from mmcv.ops import MultiScaleDeformableAttention
print("MMCV CUDA ops loaded successfully")
EOF


✅ If this prints without error → you’re on the correct path

❌ If this fails → do NOT continue

STEP 5 — Install MMPose
pip install -U mmpose

If you see pip related issues here.
ModuleNotFoundError: No module named 'pip'
Failed to build 'chumpy'

conda activate env_mmpose_env
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

Pre install Chumpy(Critical trick)

pip install chumpy --no-build-isolation

Then again try  
 pip install -U mmpose


Verify:

python - <<EOF
import mmpose
print("MMPose version:", mmpose.__version__)
EOF

STEP 6 — Verify Registry (the Mac killer test)

Run:

python - <<EOF
from mmpose.models import TopdownPoseEstimator
print("MMPose registry OK")
EOF


If this passes →
🎯 You are officially past 90% of setup failures

STEP 7 — Test MIM CLI
mim --version
mim list


You should see mmpose listed.

### NumPy / OpenCV Compatibility
MMPose (v1.3.x) is not compatible with NumPy 2.x.
Use:
- numpy==1.26.4
- opencv-python==4.8.1.78
Reinstall xtcocotools after any NumPy change.


Crtitical:
After installing mmcv
Check with python -c "import mmcv._ext"

If you find problem with mmcv._ext not found issue.
Then:
pip uninstall mmcv mmcv-full -y
# Example for CUDA 11.7 and PyTorch 2.0 (adjust versions as needed)
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

### 🔴 Important: Version Selection Rule

MMCV does NOT publish wheels for all PyTorch versions.

If pip downloads `.tar.gz` instead of `.whl`, it means:
- that Torch + CUDA combo is unsupported

Rule:
1. Pick a PyTorch version with MMCV wheels
2. Match CUDA to that PyTorch
3. Never the other way around

Known stable combo:
- torch 2.0.0 + cu117
- mmcv 2.0.1

pip install mmdet==3.3.0
pip install mmpose==1.


TO add area feild in annotations val.json.
(env_mmpose_env) chetan@chetan-MS-7D46:~/AI_First/horse_health_analysis/horse_health_analysis/mmpose$ python fix_coco_area.py data/annotations/val.json


**** Its not running fuly right now: there are path related issues
I have made an train_hoof_pose.sh file. which will automatically add missing parts and then run traning script.

mmpose$ python tools/train.py custom_configs/rtmpose_hoof_4kp.py

with 

make it runnable:

chmod +x train_hoof_pose.sh
