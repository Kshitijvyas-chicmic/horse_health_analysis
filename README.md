
# 🐴 Horse Hoof-Pastern Axis (HPA) Measurement System

A clinical-grade AI system for detecting anatomical landmarks on horse limbs and calculating **Hoof-Pastern Axis (HPA)** deviation angles from standard 2D photographs.

---

## 📐 Angle Measurement Methodology

### Reference Frames

There are two valid ways to express angles:

**Vertical Reference (Anatomical)**
- Angle measured relative to the vertical axis
- Used internally by this system for stability and consistency
- Less sensitive to camera tilt and image orientation

**Ground Reference (Device-Compatible)**
- Angle measured relative to the ground (horizontal plane)
- Commonly reported by physical farriery devices
- Conversion: `Ground Angle = 90° − Vertical Angle`

### 🔧 Why Results May Differ Slightly From Physical Devices

Physical measurement devices estimate angles using the internal bony column alignment of the limb. Image-based analysis estimates angles using the external hoof capsule and visible pastern silhouette.

Because:
- Bones are positioned slightly forward within soft tissue
- Hoof capsules can flare or wear unevenly
- Images are 2D projections of a 3D structure

A small, consistent difference (typically 3–6°) is expected and biologically normal. This system has been validated against device readings and shows high consistency, stable bias, and clinically reasonable agreement.

---

## 🧠 Model Strategy: Front-Edge vs Center-Axis

### Old Approach: Center-Axis (March 9th Model)
The model was trained with keypoints placed on the **center/bone axis** of the pastern and hoof. This was stable but produced measurements that differed from physical devices by a consistent offset. A manual "axis-to-surface shift" was applied in post-processing to compensate.

### Current Approach: Front-Edge (April 27th Model ✅)
The model is now trained with keypoints placed on the **visible front surface/edge** of the leg. This eliminates the need for post-processing corrections and directly measures the surface angle that physical devices also measure.

**Key decision:** The "axis-to-surface shift" logic in `inference_on_new_image_refined.py` is **commented out** because the Front-Edge model predicts surface points directly. If you revert to the March 9th model, you must re-enable those lines.

---

## 🚀 Model Hub (Checkpoints)

| Model | Checkpoint | Epochs | AP Score | Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **March 9 (Baseline)** | `work_dirs/rtmpose_hoof_4kp_clinical_stabilization/epoch_300.pth` | 300 | ~0.67 | Center-Axis |
| **April 24** | `work_dirs/rtmpose_hoof_manual_24_april/epoch_300.pth` | 300 | ~0.67 | Front-Edge (loose bbox) |
| **April 27 (Current ✅)** | `work_dirs/rtmpose_hoof_manual_27_april/epoch_150.pth` | 150 | **0.795** | Front-Edge + Tight BBox (0.50 ratio) |

> **Use April 27 model** for all production inference. It is the most accurate and background-resistant model to date.

---

## 🔑 Key Engineering Decisions (April 27th)

### 1. Tight Scanner Ratio (MODEL_RATIO = 0.50)
**Problem:** The original inference scanner used a `192/256 = 0.75` aspect ratio (Width/Height). This wide crop included background fences and building rails, which confused the AI — confidence was only ~0.46.

**Fix:** Reduced `MODEL_RATIO` to `0.50` in `inference_on_new_image_refined.py`. This "narrows" the crop around just the leg, effectively blinding the AI to background distractions.

**Result:** Confidence jumped from 0.46 → **1.30** on the same images. No retraining required.

### 2. Tight Training BBoxes (Aspect Ratio Enforcement)
**Problem:** Training data bounding boxes had inconsistent aspect ratios, teaching the AI to "expect" wide crops with background noise.

**Fix:** Added aspect ratio enforcement (target 0.50) to `fix_bbox_from_keypoints.py`. Every training crop is now forced into a narrow portrait orientation matching what the inference scanner shows the model.

**Rule:** Always run `fix_bbox_from_keypoints.py` before training. Train how you Test.

### 3. Brightness/Contrast Augmentation
**Problem:** Dark horses in shadowy stalls caused the AI to hallucinate the hoof-ground boundary, placing the bottom point in the shadow instead of on the hoof capsule.

**Fix:** Added `RandomBrightnessContrast`, `HueSaturationValue`, and `CLAHE` augmentations to `rtmpose_hoof_4kp_copy.py`. During training, the AI now sees artificially brightened and darkened versions of each image, teaching it to "see through" shadows.

### 4. AP Score Improvement Summary

| Epoch | April 24 AP | April 27 AP | Gain |
|:------|:------------|:------------|:-----|
| 10 | 0.341 | **0.510** | +50% |
| 70 | 0.635 | **0.769** | +21% |
| 110 | 0.596 | **0.773** | +30% |
| 130 | 0.673 (peak) | **0.795** (peak) | **+18%** |

---

## 🛠️ Full Training Workflow

### Step 1: Export Annotations from CVAT
Export your annotations as COCO JSON. Keep all images in a common folder.

### Step 2: Split into Train / Val
```bash
python3 split_dataset.py \
  --input data/annotations/cvat/your_annotations.json \
  --train_out data/annotations/train.json \
  --val_out data/annotations/val.json \
  --ratio 0.85
```

### Step 3: Fix & Tighten Bounding Boxes ⚠️ ALWAYS DO THIS
```bash
# From the project root:
conda activate env_mmpose_env
python fix_bbox_from_keypoints.py data/annotations/train.json
python fix_bbox_from_keypoints.py data/annotations/val.json
# Output: train_fixed.json and val_fixed.json
```
This enforces the **0.50 aspect ratio** on all bounding boxes, matching the inference scanner.

### Step 4: Start Training
```bash
cd mmpose
python tools/train.py custom_configs/rtmpose_hoof_4kp_copy.py
```

### Step 5: Resume Training (No Need to Start From Zero!)
If you want to train beyond the initial `max_epochs`, simply update the config and resume:
```bash
# 1. Change max_epochs in rtmpose_hoof_4kp_copy.py to your new target (e.g., 300)
# 2. Run with --resume flag:
python tools/train.py custom_configs/rtmpose_hoof_4kp_copy.py --resume
```
The model will pick up exactly from the last saved checkpoint.

---

## 🔍 Running Inference

### On a Single Image
```bash
conda activate env_mmpose_env
cd mmpose
python3 demo/inference_on_new_image_refined.py /path/to/image.jpg \
  --checkpoint work_dirs/rtmpose_hoof_manual_27_april/epoch_150.pth \
  --out-file result.jpg
```

### Understanding the Scanner Output
The inference script evaluates **8 scanning zones** (4 base zones × 2 horizontal offsets):

| Zone | Purpose |
|:-----|:--------|
| `Floor-Scan` | Looks at the bottom 60% of the image (hoof + floor area) |
| `Anatomy-Scan` | Looks at the middle pastern anatomy zone |
| `Top-Anatomy` | Looks at the upper pastern / cannon area |
| `Global-Scan` | Full-frame fallback scan |

Each zone is scanned with horizontal offsets (`off=0`, `off=-0.15`, `off=+0.15`) to handle legs that are off-center in the frame. The zone with the **highest Score** (Avg_Conf × keypoint quality) wins.

**Healthy confidence score:** > 1.0 means the model is very certain. Scores below 0.5 indicate the AI is struggling (dark horse, complex background).

### What the Output Lines Mean
```
Pastern: 60.9 | Hoof: 50.9 | Dev: 10.1
```
- **Pastern**: Angle of the pastern line from vertical (degrees)
- **Hoof**: Angle of the hoof wall from vertical (degrees)
- **Dev**: |Pastern − Hoof| — The clinical HPA deviation. Ideal = < 5°. > 10° = clinical concern.

---

## 🏋️ Training Configuration Reference

Key parameters in `mmpose/custom_configs/rtmpose_hoof_4kp_copy.py`:

| Parameter | Current Value | Notes |
|:----------|:-------------|:------|
| `max_epochs` | 150 | Sweet spot for ~585 images. Use 300 for 900+ images. |
| `input_size` | `(192, 256)` | Portrait mode — matches 0.50 W/H ratio |
| `rotate_factor` | 10 | ±10° geometric invariance |
| `RandomBrightnessContrast` | ✅ Enabled | Handles dark horses / shadowy stalls |
| `HueSaturationValue` | ✅ Enabled | Handles coat color variation |
| `CLAHE` | ✅ Enabled | Enhances edge visibility in low contrast |
| Cosine LR decay | begins epoch 75 | Fine-tuning phase for 150-epoch runs |

---

## 📊 Dataset Information

| Version | Images | Format | Notes |
|:--------|:-------|:-------|:------|
| v7 | 585 | COCO Keypoints | Current training set (train: 468, val: 117) |
| Target | 900+ | COCO Keypoints | Add dark horses, complex backgrounds, odd angles |

### 4 Keypoints per Image:
1. **Pastern Top** (Red) — Top of pastern front edge
2. **Pastern Bottom** (Orange) — Bottom of pastern / coronary band level
3. **Hoof Top** (Green) — Top of hoof capsule front wall
4. **Hoof Bottom** (Blue) — Toe tip / ground contact point

---

## 🌍 Environment Setup

### Local Machine (Recommended)
```bash
# Install conda
conda create -n env_mmpose_env python=3.10 -y
conda activate env_mmpose_env

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install MMCV for Python 3.10 + CUDA 12.1
pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl

# Install remaining dependencies
pip install mmengine mmdet albumentations==1.3.1
```

### Google Colab (CUDA 12.1, Python 3.12)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Use the Python 3.12 specific MMCV wheel:
pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/mmcv-2.2.0-cp312-cp312-manylinux1_x86_64.whl
pip install mmengine mmdet albumentations==1.3.1
```

---

## 🚀 API & Production Deployment

### Start the Server
```bash
# Development
uvicorn apis.main:app --host 0.0.0.0 --port 8001

# Production (Gunicorn)
gunicorn apis.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 120
```

### API Endpoints
| Version | Endpoint | Description |
| :--- | :--- | :--- |
| **v2** | `/api/v2/analyze` | Stable MMPose. Returns flat JSON. |
| **v3** | `/api/v3/analyze` | Clinical Dual Model. MMPose + YOLO with granular angles. |

Interactive Docs: `http://<server-ip>:8001/docs`

---

## ⚠️ Important Notes

1. **Never `git add .` on the main directory.** Add `mmpose/` changes separately. The mmpose submodule has custom code changes in `mmpose/datasets/transforms/common_transforms.py` — preserve these on every mmpose update.

2. **The axis-shift code is commented out.** Lines in `inference_on_new_image_refined.py` that mathematically shift points from bone-center to surface are intentionally commented out. The Front-Edge model predicts surface points directly. Only re-enable if you switch back to a Center-Axis model.

3. **Always re-run `fix_bbox_from_keypoints.py` before retraining** after adding new images. The aspect ratio enforcement ensures train/test consistency.

4. **Best checkpoint selection:** Monitor AP@0.75 in the training logs (`scalars.json`). The best checkpoint is typically NOT the last one — check epoch 110–130 range for peak precision.