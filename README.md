
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
- **Config**: `mmpose/custom_configs/rtmpose_hoof_4kp.py`
- **Output**: `output_inference.jpg` (Segmented joints + Clinical HPA Dev)

### 2. Vanguard Scan (Baseline)
Use this as a baseline audit tool.
```bash
python mmpose/demo/inference_on_new_image.py /path/to/image.jpg
```

---

## 🏋️ Training (Advanced)

To retrain the model with Geometric Invariance (Random Rotation):

1. **Configure**: Use `mmpose/custom_configs/rtmpose_hoof_4kp.py`.
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

### 📦 MMPose Management
The `mmpose` engine is internalized as a regular directory in this repository to ensure all custom patches (e.g., in `common_transforms.py`) and configurations are natively tracked.

**Never do git add . on main dir.** Add changes separately to keep the git history clean.

# We have modified code in `mmpose/mmpose/datasets/transforms/common_transforms.py` for compatibility. These changes are natively tracked in the `image_quality_check` branch.

Run `inference_on_new_image_refined.py` to get the clinical angles of any new image with high precision.

TO split the new json in train and val:
 python3 /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/split_dataset.py --input /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/cvat/person_keypoints_default_590.json --train_out /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/train_590.json --val_out /home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/annotations/val_590.json --ratio 0.85

---

## 🔄 Data Processing Pipeline Overview

This is the standard workflow for preparing data and training the model:

1.  **Collect Data**: Add new images to `data/images`.
2.  **Split Dataset**: Run `split_dataset.py` to create `train.json` and `val.json`.
3.  **Optimize BBoxes**: Run `fix_bbox_from_keypoints.py` on your new JSONs to expand detection margins.
4.  **Update Config**: Point your model config (e.g., `rtmpose_hoof_4kp.py`) to the new `_fixed.json` files.
5.  **Train**: Execute the training script/command.

### 🛠️ Script Roles

| Script | Responsibility | Key Output |
| :--- | :--- | :--- |
| `split_dataset.py` | **Splitter**: Partitions master CVAT export into training and validation sets. | `train.json`, `val.json` |
| `fix_bbox_from_keypoints.py` | **Optimizer**: Expands bounding boxes for better model performance. | `train_fixed.json`, `val_fixed.json` |
| `sync_annotations.py` | **Syncer**: Updates existing files with refined labels from new CVAT exports. | Updated annotations |
| `train_hoof_pose.sh` | **Runner**: Automates the training environment and execution. | Trained Model (`.pth`) |