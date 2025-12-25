python3 -m venv mmpose_env
source mmpose_env/bin/activate

Upgrade basics:
pip install --upgrade pip setuptools wheel

NO GPU => CUDA available, although project will work for this project usng CPU but for better perofmance and results on large scale!!There are diferent cloud options:

options:
Cloud GPU (recommended for you)

Best options:

Google Colab (Free / Pro) → easiest

Kaggle Notebooks

AWS / GCP / Paperspace

Pytorch for mmpose -> pytorch will not work with python 3.11+ so I have to install 3:10 for mmpose env.
/usr/local/bin/python3.10 -m venv env_mmpose

pip install torch torchvision torchaudio

Install MMEngine + MMCV (required by MMpose)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

Install MMpose
mim install mmpose


## MMpose Environment Notes

MMpose currently requires NumPy < 2 due to binary compatibility
with PyTorch, MMCV, and Chumpy.

Do NOT upgrade NumPy to 2.x in env_mmpose.

Known working setup:
- Python 3.10
- NumPy 1.26.x
- PyTorch 2.x (CPU)
- MMCV via openmim
- MMpose 1.3.x

3.1 Model choice (critical)
Why NOT top-down full body models

You have 4 keypoints

Single limb

Tight crop

Side view only

Using HRNet/W48 is overkill and unstable with small data.


To train the model mmpose
✅ Recommended model (best for your case)

RTMPose-Tiny (Top-Down)

Why:

Works extremely well with small keypoint counts

Fast convergence

Designed for custom skeletons

Robust to limited data

✔ Industry-grade
✔ Small-dataset friendly
✔ Future-proof


To run the traning script:
mim train mmpose ./mmpose/configs/hoofs_keypoints/rtmpose_hoof_4kp.py
