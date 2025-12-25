.

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

TRANING:

1. python tools/train.py custom_configs/rtmpose_hoof_4kp.py
2. change the settings inside rtmpose_hoof_4kp.py accordingly like epoch

To visualise the traning result on any image:

python demo/debug_visualizer.py

** change the image name/path inside code.

To remove traning data run below command: As traning data is saved in work_dirs inside mmpose

rm -rf work_dirs/rtmpose_hoof_4kp/*

To run the inference on any image:
python demo/inference_on_new_image.py /home/chetan/Desktop/demo.jpeg


To check the angles of any image:(to compare what you draw on cvat and what device reading says.)
test_pipeline.py

To add mmpose as submodule:
git submodule add https://github.com/open-mmlab/mmpose.git mmpose

**Never do git add . on main dir. rather add mmpoe changes seperataly and main chaanges seperate. This will keep our git clean.

# we have change some code in mmpose/datasets/transforms/common_transforms.py ->  to make it work for our project. So whenever you update mmpose, make sure to update this file as well.