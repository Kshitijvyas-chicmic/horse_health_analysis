#!/usr/bin/env python
import cv2
import numpy as np
import os
import sys
import math
from argparse import ArgumentParser

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# --- HPA Logic (Synced with test_pipeline.py) ---
def angle_from_vertical(v):
    # Vertical reference = (0, -1) because Y increases downward
    vertical = (0, -1)
    dot = v[0]*vertical[0] + v[1]*vertical[1]
    det = v[0]*vertical[1] - v[1]*vertical[0]
    return math.degrees(math.atan2(det, dot))

def clinical_angle(image_angle):
    # Converts angle-from-vertical to angle-from-ground
    angle = abs(image_angle) % 180
    if angle > 90:
        angle = 180 - angle
    return 90 - angle

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    #already a stable and tested file
    parser.add_argument('--config', default='custom_configs/rtmpose_hoof_4kp.py', help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/rtmpose_hoof_4kp_portrait/epoch_100.pth', help='Checkpoint file')



    #file added with refinement changes. If need we can add or remove it depdnding on our results
    #parser.add_argument('--config', default='custom_configs/rtmpose_hoof_4kp.py', help='Config file')
    #parser.add_argument('--checkpoint', default='work_dirs/rtmpose_hoof_4kp_clinical_stabilization/epoch_300.pth', help='Checkpoint file')

    
    parser.add_argument('--out-file', default='output_inference.jpg', help='Output image file')
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    args = parser.parse_args()

    register_all_modules()

    # 1. Initialize Model
    print("Initializing model...")
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.flip_test = True
    model = init_model(cfg, args.checkpoint, device=args.device)

    # 2. Load Image
    img_path = args.img
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    MODEL_RATIO = 192 / 256  # 0.75

    # 4. MULTI-SCAN ENSEMBLE (The "Vanguard" Strategy)
    # Total coverage with 4 overlapping zones to ensure 100% capture.
    zones = [
        {'name': 'Floor-Scan',    'y1': int(img_h * 0.4), 'y2': img_h},
        {'name': 'Anatomy-Scan',  'y1': int(img_h * 0.2), 'y2': int(img_h * 0.8)},
        {'name': 'Top-Anatomy',   'y1': 0,               'y2': int(img_h * 0.6)},
        {'name': 'Global-Scan',   'y1': 0,               'y2': img_h}
    ]
    
    best_results = None
    best_agg_score = -1
    best_zone_name = "None"
    
    print(f"🔍 SCANNINC: Evaluating {len(zones)} zones for anatomy...")
    
    for zone in zones:
        z_h = zone['y2'] - zone['y1']
        z_w = z_h * MODEL_RATIO
        
        # Center horizontally
        z_x1 = max(0, (img_w - z_w) / 2)
        z_x2 = min(img_w, z_x1 + z_w)
        z_bbox = np.array([z_x1, zone['y1'], z_x2, zone['y2']], dtype=np.float32)
        
        # Inference
        results = inference_topdown(model, img, bboxes=z_bbox[None, :])
        res = results[0]
        scores = res.pred_instances.keypoint_scores[0]
        
        # Aggregate score logic: We want ALL points to be found.
        # We also penalize points with extremely low confidence (< 0.1)
        agg_score = np.mean(scores) * 10
        if any(s < 0.15 for s in scores): agg_score -= 5
        
        print(f"  - [{zone['name']}]: Score={agg_score:.2f} (Avg_Conf={np.mean(scores):.2f})")
        
        if agg_score > best_agg_score:
            best_agg_score = agg_score
            best_results = res
            best_zone_name = zone['name']

    # 5. Process Winner
    keypoints = best_results.pred_instances.keypoints[0]
    scores = best_results.pred_instances.keypoint_scores[0]
    
    # Visualize
    vis_img = img.copy()
    
    # 0:Red, 1:Orange, 2:Green, 3:Blue
    clrs = [(0,0,255), (0,165,255), (0,255,0), (255,0,0)]
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(vis_img, (int(x), int(y)), 6, clrs[i], -1)
        cv2.putText(vis_img, f"{scores[i]:.2f}", (int(x)+5, int(y)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, clrs[i], 1)

    # 6. HPA Analysis
    REJECT_THRESHOLD = 0.28 # High enough to block noise, low enough to accept distant anatomy
    if all(s > REJECT_THRESHOLD for s in scores):
        pts = {i: list(keypoints[i]) for i in range(4)}
        
        # Direction Normalization (Safety Invariant)
        if pts[3][0] < pts[0][0]:
            center_x = (pts[0][0] + pts[3][0]) / 2
            for i in range(4):
                pts[i][0] = 2 * center_x - pts[i][0]
        
        v_pastern = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        v_hoof = (pts[3][0] - pts[2][0], pts[3][1] - pts[2][1])
        
        p_angle = clinical_angle(angle_from_vertical(v_pastern))
        h_angle = clinical_angle(angle_from_vertical(v_hoof))
        diff = abs(p_angle - h_angle)
 
        # Visualization Lines
        cv2.line(vis_img, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[1][0]), int(keypoints[1][1])), (255, 128, 0), 2)
        cv2.line(vis_img, (int(keypoints[2][0]), int(keypoints[2][1])), (int(keypoints[3][0]), int(keypoints[3][1])), (0, 128, 255), 2)
 
        print(f"\n✅ HPA SUCCESS (Zone: {best_zone_name}):")
        print(f"  Pastern: {p_angle:.1f}° | Hoof: {h_angle:.1f}° | Dev: {diff:.1f}°")
 
        color = (0, 255, 0) if diff < 3 else (0, 0, 255)
        cv2.putText(vis_img, f"HPA Dev: {diff:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        print(f"\n❌ REJECTED: Low confidence in best zone ({best_zone_name}).")
        cv2.putText(vis_img, "REJECTED: Low Confidence", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(args.out_file, vis_img)
    print(f"Result saved to {args.out_file}")

if __name__ == '__main__':
    main()
