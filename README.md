
📐 Angle Measurement Methodology

This system estimates hoof and pastern angles from 2D images using anatomical landmarks annotated on the horse’s limb.

Reference Frames

There are two valid ways to express angles:

Vertical Reference (Anatomical)

Angle measured relative to the vertical axis

Used internally by this system for stability and consistency

Less sensitive to camera tilt and image orientation

Ground Reference (Device-Compatible)

Angle measured relative to the ground (horizontal plane)

Commonly reported by physical farriery devices

The two are related by:

Ground Angle = 90° − Vertical Angle

🔧 Why Results May Differ Slightly From Devices

Physical measurement devices estimate angles using the internal bony column alignment of the limb.

Image-based analysis estimates angles using the external hoof capsule and visible pastern silhouette.

Because:

bones are positioned slightly forward within soft tissue

hoof capsules can flare or wear unevenly

images are 2D projections of a 3D structure

A small, consistent difference (typically 3–6°) is expected and biologically normal.

This system has been validated against device readings and shows:

high consistency

stable bias

clinically reasonable agreement

✅ Intended Use

This tool is designed for:

comparative analysis (before / after shoeing)

trend tracking over time

decision support for hoof balance

It is not a replacement for invasive or radiographic measurement.

4️⃣ Why this is the correct professional approach

✔ You did not “force” results to match
✔ You documented reference frames
✔ You preserved explainability
✔ You gave the client transparency

This is how medical, biomechanical, and sports-analysis software is shipped.


CONDA installer:
#we are using conda instead ov venv as it works well with CUDA.
Make sure to install conda in your machine.
Then: 
check conda --version
conda create -n env_mmpose_env python=3.10 -y
conda activate env_mmpose_env

We will install PyTorch CUDA 12.1, which is fully compatible.

Run this inside env_mmpose_env:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


TO train the images:
1. get annotation json from cvat from required images.
2. Split images and json by 80-20 ratio (80% in training and 20% in validation)
3. Folder data-> images a. val b. train
4. Folder data -> annotaions - a. val.json b. train.json
5. Main json to put in data -> annotations -> cvat.
6. Make sure there shall be same image id in both images and annotations key in json.

After we seperate the json
1. We are increasing the bbox area by.
python fix_bbox_from_keypoints.py data/annotations/train.json
python fix_bbox_from_keypoints.py data/annotations/val.json

this will make new json with name val_fixed.json and train_fixed.json

This is required as our curent bbox is tight and very close to the data points hence making difficult for rtmpose traning.

---

## 🚀 Model Hub (Checkpoints)

We provide two distinct models depending on your clinical requirements:

| Model | Checkpoint | Training | Best For |
| :--- | :--- | :--- | :--- |
| **Baseline** | `work_dirs/rtmpose_hoof_4kp_portrait/epoch_100.pth` | 100 Epochs | Super-expert precision on perfectly leveled photos. |
| **Stabilized** | `work_dirs/rtmpose_hoof_4kp_clinical_stabilization/epoch_300.pth` | 300 Epochs | **Clinical Calibration**: Detects real pathologies and handles image tilt (10° rotation invariant). |

---

## 🛠️ Inference Tools

### 1. Production Inference (Recommended)
Use this for final clinical reports. It features calibrated math and zero-overshoot visualization.
```bash
python demo/inference_on_new_image_refined.py /path/to/image.jpg
```
- **Config**: `custom_configs/rtmpose_hoof_4kp_copy.py`
- **Output**: `output_inference.jpg` (Segmented joints + Clinical HPA Dev)

### 2. Vanguard Scan (Baseline)
Use this as a baseline audit tool.
```bash
python demo/inference_on_new_image.py /path/to/image.jpg
```

---

## 🏋️ Training (Advanced)

To retrain the model with Geometric Invariance (Random Rotation):

1. **Configure**: Use `custom_configs/rtmpose_hoof_4kp_copy.py`.
2. **Parameters**:
   - `rotate_factor=10`: Decouples anatomy from image orientation (Geometric Invariance).
   - `max_epochs=300`: Allows the joint localization to "settle" for clinical precision.
3. **Run**:
```bash
python tools/train.py custom_configs/rtmpose_hoof_4kp_copy.py --work-dir work_dirs/rtmpose_hoof_4kp_clinical_stabilization
```

To visualise the traning result on any image:

python demo/debug_visualizer.py

** change the image name/path inside code.

To remove traning data run below command: As traning data is saved in work_dirs inside mmpose

rm -rf work_dirs/rtmpose_hoof_4kp/*

To run the baseline inference:
python demo/inference_on_new_image.py /home/chetan/Desktop/demo.jpeg


To check the angles of any image:(to compare what you draw on cvat and what device reading says.)
test_pipeline.py

To add mmpose as submodule:
git submodule add https://github.com/open-mmlab/mmpose.git mmpose

**Never do git add . on main dir. rather add mmpose changes seperataly and main chaanges seperate. This will keep our git clean.

# we have change some code in mmpose/datasets/transforms/common_transforms.py ->  to make it work for our project. So whenever you update mmpose, make sure to update this file as well.

Run `inference_on_new_image_refined.py` to get the clinical angles of any new image with high precision.

TO split the new json in train and val:
 python3 /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/split_dataset.py --input /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/cvat/person_keypoints_default_590.json --train_out /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/train_590.json --val_out /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/val_590.json --ratio 0.85

---

## 🚀 API & Production Deployment (v3)

A production-grade AI system for detecting anatomical landmarks on horse limbs and calculating **Hoof-Pastern Axis (HPA)** metrics. This system uses both **MMPose** and **YOLOv8** to provide clinical-grade precision.

### 1. Environment Setup (DevOps)
The API is optimized for **CPU-only** production servers.

```bash
# Create Conda Environment
conda create -n horse_health python=3.10 -y
conda activate horse_health

# Install Production Dependencies (CPU Optimized)
pip install -r requirements.txt
```

### 2. Verify Model Checkpoints
Ensure the following weights are present in the project root:
- **MMPose**: `mmpose/work_dirs/rtmpose_hoof_unified_jan12/epoch_300.pth`
- **YOLO Medium**: `runs/segment/hpa_v8m_full_v1/weights/best.pt`

### 3. Start the Server
For production, use **Gunicorn** with **Uvicorn** workers:
```bash
# Production server (Port 8000)
gunicorn apis.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 120

# Development/Sharing (Port 8001)
uvicorn apis.main:app --host 0.0.0.0 --port 8001
```

### 4. API Versioning
| Version | Endpoint | Description |
| :--- | :--- | :--- |
| **v2** | `/api/v2/analyze` | **Stable (MMPose)**. Returns original flat JSON schema. |
| **v3** | `/api/v3/analyze` | **Clinical (Dual Model)**. Returns MMPose + YOLO with granular angles. |

---

## 🧪 Clinical Logic (Dual Model)
Both models implement **Anatomical Sanity Filters**. An image is rejected (`success=False`) if:
1.  **Fetlock** is below the **Coronary Band**.
2.  **Coronary Band** is below the **Toe**.
3.  **Model Confidence** is low (<0.4).

Interactive Documentation (Swagger) is available at: `http://<server-ip>:8001/docs`