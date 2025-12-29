import json
import os
import cv2
import numpy as np
import math
import csv
import sys
sys.path.append('mmpose')
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# --- HPA Calculation Logic (Copied from debug_visualizer.py) ---
def angle_from_vertical(v):
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

def calculate_hpa_metrics(keypoints):
    # keypoints is [4, 2]
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
    return p_angle, h_angle, abs(p_angle - h_angle)

def main():
    register_all_modules()
    config = 'mmpose/custom_configs/rtmpose_hoof_4kp.py'
    checkpoint = 'mmpose/work_dirs/rtmpose_hoof_4kp_portrait/epoch_100.pth'
    
    print("🚀 Initializing model for evaluation...")
    from mmengine.config import Config
    cfg = Config.fromfile(config)
    cfg.model.test_cfg.flip_test = True
    model = init_model(cfg, checkpoint, device='cpu')
    
    val_json = 'data/annotations/val_fixed.json'
    img_dir = 'data/images/val'
    with open(val_json, 'r') as f:
        data = json.load(f)
    
    results = []
    print(f"📊 Evaluating Accuracy on {len(data['images'])} validation images...")
    
    debug_dir = 'debug_eval'
    os.makedirs(debug_dir, exist_ok=True)
    
    for idx, img_info in enumerate(data['images']):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        
        # Get GT keypoints and bbox
        ann = [a for a in data['annotations'] if a['image_id'] == img_id][0]
        gt_kpts = np.array(ann['keypoints']).reshape(-1, 3)[:, :2]
        gt_p, gt_h, gt_dev = calculate_hpa_metrics(gt_kpts)
        
        bbox = np.array(ann['bbox'], dtype=np.float32)
        # Convert [x1, y1, w, h] to [x1, y1, x2, y2]
        bbox_xyxy = bbox.copy()
        bbox_xyxy[2:] += bbox_xyxy[:2]
        
        # Inference
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        res = inference_topdown(model, img, bboxes=bbox_xyxy[None, :])[0]
        pred_kpts = res.pred_instances.keypoints[0]
        pred_p, pred_h, pred_dev = calculate_hpa_metrics(pred_kpts)
        
        err_p = abs(gt_p - pred_p)
        err_h = abs(gt_h - pred_h)
        err_dev = abs(gt_dev - pred_dev)

        # Draw Debug Images for ALL validation samples
        if True:
            vis = img.copy()
            # 0:Red, 1:Orange, 2:Green, 3:Blue
            clrs = [(0,0,255), (0,165,255), (0,255,0), (255,0,0)]
            
            # Draw GT with large circles and lines
            for i, p in enumerate(gt_kpts):
                cv2.circle(vis, (int(p[0]), int(p[1])), 10, clrs[i], -1)
            cv2.line(vis, (int(gt_kpts[0][0]), int(gt_kpts[0][1])), 
                     (int(gt_kpts[1][0]), int(gt_kpts[1][1])), (255, 255, 255), 2)
            cv2.line(vis, (int(gt_kpts[2][0]), int(gt_kpts[2][1])), 
                     (int(gt_kpts[3][0]), int(gt_kpts[3][1])), (255, 255, 255), 2)
            
            # Draw Pred with small circles and lines
            for i, p in enumerate(pred_kpts):
                cv2.circle(vis, (int(p[0]), int(p[1])), 6, clrs[i], 2)
            cv2.line(vis, (int(pred_kpts[0][0]), int(pred_kpts[0][1])), 
                     (int(pred_kpts[1][0]), int(pred_kpts[1][1])), (0, 255, 255), 2)
            cv2.line(vis, (int(pred_kpts[2][0]), int(pred_kpts[2][1])), 
                     (int(pred_kpts[3][0]), int(pred_kpts[3][1])), (255, 255, 0), 2)
            
            cv2.putText(vis, f"GT Dev: {gt_dev:.1f} Pred: {pred_dev:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, f"idx_{file_name}"), vis)

        results.append({
            'Image': file_name,
            'GT_Pastern_Angle': round(gt_p, 2),
            'Pred_Pastern_Angle': round(pred_p, 2),
            'Pastern_Error': round(err_p, 2),
            'GT_Hoof_Angle': round(gt_h, 2),
            'Pred_Hoof_Angle': round(pred_h, 2),
            'Hoof_Error': round(err_h, 2),
            'GT_Deviation': round(gt_dev, 2),
            'Pred_Deviation': round(pred_dev, 2),
            'Deviation_Error': round(err_dev, 2)
        })
    
    # Save CSV
    out_csv = 'hpa_accuracy_report.csv'
    keys = results[0].keys()
    with open(out_csv, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    # Summary Calculations
    total = len(results)
    avg_err_p = sum(r['Pastern_Error'] for r in results) / total
    avg_err_h = sum(r['Hoof_Error'] for r in results) / total
    
    # Percent of samples where deviation error is < 3.0 degrees
    tolerance = 3.0
    accurate_samples = sum(1 for r in results if r['Deviation_Error'] <= tolerance)
    accuracy_percent = (accurate_samples / total) * 100
    
    print("\n" + "="*40)
    print("📈 ACCURACY REPORT SUMMARY")
    print("="*40)
    print(f"Total Samples Tested:     {total}")
    print(f"Avg Pastern Angle Error:  {avg_err_p:.2f}°")
    print(f"Avg Hoof Angle Error:     {avg_err_h:.2f}°")
    print(f"Success Rate (Error < 3°): {accuracy_percent:.1f}%")
    print(f"Full Report Saved to:     {out_csv}")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()
