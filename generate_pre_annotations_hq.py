import json
import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add mmpose to path
sys.path.append('mmpose')
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

def main():
    register_all_modules()
    
    # Config and Checkpoint
    config = 'mmpose/custom_configs/rtmpose_hoof_4kp_copy.py'
    checkpoint = 'mmpose/work_dirs/rtmpose_hoof_refined_6jan_v2_rotation/epoch_300.pth'
    
    # Paths
    base_json_path = 'data/annotations/cvat/person_keypoints_default_6_jan.json'
    img_dir = 'data/images/hq_consolidation_550'
    out_json_path = 'data/annotations/cvat/hq_consolidation_pre_labeled_jan7.json'
    
    # Load Model
    print("🚀 Initializing Two-Pass Precision Engine...")
    model = init_model(config, checkpoint, device='cpu')
    
    with open(base_json_path, 'r') as f:
        coco = json.load(f)
    
    existing_images = coco['images']
    existing_anns = coco['annotations']
    
    max_img_id = max(img['id'] for img in existing_images) if existing_images else 0
    max_ann_id = max(ann['id'] for ann in existing_anns) if existing_anns else 0
    
    all_files = sorted([f for f in os.listdir(img_dir) if f.startswith('h') and f.endswith('.jpeg')])
    new_files = [f for f in all_files if int(f[1:-5]) >= 159]
    
    print(f"📊 Processing {len(new_files)} images...")
    
    for i, fname in enumerate(new_files):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None: continue
            
        h, w = img.shape[:2]
        img_id = max_img_id + 1 + i
        
        # 1. Update Image Entry
        coco['images'].append({
            "id": img_id, "width": w, "height": h, "file_name": fname,
            "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0
        })
        
        # --- PASS 1: Low-Res Detection ---
        # Resize to 512px max dim for fast detection
        scale = 512.0 / max(h, w)
        img_low = cv2.resize(img, (int(w * scale), int(h * scale)))
        lh, lw = img_low.shape[:2]
        
        # Run AI on full low-res image
        bbox_low = np.array([0, 0, lw, lh], dtype=np.float32)
        res_low = inference_topdown(model, img_low, bboxes=bbox_low[None, :])[0]
        
        coco_kpts = [0] * 12
        num_kpts = 0
        
        if len(res_low.pred_instances.keypoints) > 0:
            kpts_low = res_low.pred_instances.keypoints[0]
            scores_low = res_low.pred_instances.keypoint_scores[0]
            
            # Define a high-res crop around the detected points
            # Get min/max in global scale
            kpts_global = kpts_low / scale
            x_min, y_min = np.min(kpts_global, axis=0)
            x_max, y_max = np.max(kpts_global, axis=0)
            
            # Expand by 50% for context
            cw, ch = (x_max - x_min), (y_max - y_min)
            x_min = max(0, x_min - cw * 0.25)
            y_min = max(0, y_min - ch * 0.25)
            x_max = min(w, x_max + cw * 0.25)
            y_max = min(h, y_max + ch * 0.25)
            
            # --- PASS 2: High-Res Refinement ---
            zx, zy, zw, zh = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
            if zw > 10 and zh > 10:
                crop = img[zy:zy+zh, zx:zx+zw]
                crop_bbox = np.array([0, 0, zw, zh], dtype=np.float32)
                res_high = inference_topdown(model, crop, bboxes=crop_bbox[None, :])[0]
                
                if len(res_high.pred_instances.keypoints) > 0:
                    kpts_high = res_high.pred_instances.keypoints[0]
                    coco_kpts = []
                    for j in range(4):
                        gx = min(max(0, kpts_high[j][0] + zx), w - 1)
                        gy = min(max(0, kpts_high[j][1] + zy), h - 1)
                        coco_kpts.extend([round(float(gx), 2), round(float(gy), 2), 2])
                    num_kpts = 4
            elif num_kpts == 0:
                # Fallback to Pass 1 results if Pass 2 failed but Pass 1 had something
                coco_kpts = []
                for j in range(4):
                    gx = min(max(0, kpts_global[j][0]), w - 1)
                    gy = min(max(0, kpts_global[j][1]), h - 1)
                    coco_kpts.extend([round(float(gx), 2), round(float(gy), 2), 2])
                num_kpts = 4
        
        # If still nothing, use center cluster as requested by user's "at least save level selection"
        if num_kpts == 0:
            cx, cy = w // 2, h // 2
            offset = 100
            coco_kpts = [
                cx - offset, cy - offset, 2,
                cx + offset, cy - offset, 2,
                cx - offset, cy + offset, 2,
                cx + offset, cy + offset, 2
            ]
            num_kpts = 4

        # 3. Annotation Entry
        max_ann_id += 1
        coco['annotations'].append({
            "id": max_ann_id, "image_id": img_id, "category_id": 1,
            "segmentation": [], "area": round(float(w * h * 0.1), 2),
            "bbox": [0, 0, w, h], "iscrowd": 0,
            "attributes": {"occluded": False, "keyframe": True},
            "keypoints": coco_kpts, "num_keypoints": num_kpts
        })
        
        if (i+1) % 50 == 0:
            print(f"✅ Refined {i+1}/{len(new_files)} images...")

    with open(out_json_path, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"✨ DONE! Two-pass results saved to: {out_json_path}")

if __name__ == '__main__':
    main()
