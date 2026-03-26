#!/usr/bin/env python
import cv2
import numpy as np
import os
import math
from argparse import ArgumentParser
import sys
# Allow running from project root or mmpose folder
if os.path.basename(os.getcwd()) != 'mmpose':
    if os.path.exists('mmpose'):
        sys.path.append('mmpose')

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmengine.config import Config


# =========================
# ANGLE MATH (UNCHANGED)
# =========================
def angle_from_vertical(v):
    vertical = (0, -1)
    dot = v[0]*vertical[0] + v[1]*vertical[1]
    det = v[0]*vertical[1] - v[1]*vertical[0]
    return math.degrees(math.atan2(det, dot))


def clinical_angle(image_angle):
    angle = abs(image_angle) % 180
    if angle > 90:
        angle = 180 - angle
    return 90 - angle


# =========================
# SAFE LINE DRAWING (FIX)
# =========================
def draw_angle_line(img, p_start, p_end, color, scale=1.5, thickness=3):
    p_start = np.array(p_start, dtype=np.float32)
    p_end = np.array(p_end, dtype=np.float32)

    v = p_end - p_start
    length = np.linalg.norm(v)

    if length < 1e-3:
        return

    v = v / length
    line_len = length * scale

    p_draw_end = p_start + v * line_len

    cv2.line(
        img,
        tuple(p_start.astype(int)),
        tuple(p_draw_end.astype(int)),
        color,
        thickness,
        cv2.LINE_AA
    )

# --- ANATOMICAL OFFSET CALCULATION ---
# This assumes pts_math[0] is Red (Top) and pts_math[1] is Yellow/Orange (Bottom)
# We shift them toward the "Front" of the leg to align with the bone axis.

def apply_anatomical_offset(pts, img_w):
    # Determine which direction is 'Front' (Dorsal)
    # Usually, the hoof points toward the front. 
    # If Hoof-Point-X < Fetlock-Point-X, 'Front' is to the left.
    is_left_facing = pts[2][0] < pts[1][0]
    direction = -1 if is_left_facing else 1

    # Estimate local width (This is a heuristic, but very effective)
    # We use a small percentage of image width as a proxy for leg thickness 
    # unless you have a specific segmentation mask.
    width_estimate = img_w * 0.05 

    # Shift Red Point (Top) significantly forward (20% of estimated width)
    pts[0][0] += direction * (width_estimate * 0.25)
    
    # Shift Yellow Point (Bottom) slightly forward (10% of estimated width)
    pts[1][0] += direction * (width_estimate * 0.10)
    
    return pts
# =========================
# MAIN
# =========================
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    # Smart path resolution: if run from inside 'mmpose' dir, strip the 'mmpose/' prefix
    def fix_path(p):
        if os.path.basename(os.getcwd()) == 'mmpose' and p.startswith('mmpose/'):
            return p[7:] # strip 'mmpose/'
        return p

    parser.add_argument('--config', default=fix_path('mmpose/custom_configs/rtmpose_hoof_4kp_copy.py'))
    parser.add_argument('--checkpoint', default=fix_path('mmpose/work_dirs/rtmpose_hoof_manual_9_march/epoch_300.pth'))

    parser.add_argument('--out-file', default='output_inference.jpg')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # Apply fix to user-provided paths if they were passed manually but still use the root prefix
    args.config = fix_path(args.config)
    args.checkpoint = fix_path(args.checkpoint)

    register_all_modules()

    print(f"Initializing model from: {args.config}")
    cfg = Config.fromfile(args.config)
    cfg.model.test_cfg.flip_test = True
    model = init_model(cfg, args.checkpoint, device=args.device)

    if not os.path.exists(args.img):
        print("Image not found")
        return

    img = cv2.imread(args.img)
    img_h, img_w = img.shape[:2]
    MODEL_RATIO = 192 / 256

    zones = [
        {'name': 'Floor-Scan',   'y1': int(img_h * 0.4), 'y2': img_h},
        {'name': 'Anatomy-Scan', 'y1': int(img_h * 0.2), 'y2': int(img_h * 0.8)},
        {'name': 'Top-Anatomy',  'y1': 0,               'y2': int(img_h * 0.6)},
        {'name': 'Global-Scan',  'y1': 0,               'y2': img_h},
    ]

    best_res = None
    best_score = -1
    best_zone = None

    valid_zones_results = []
    print(f"🔍 SCANNING: Evaluating {len(zones)} zones...")
    for z in zones:
        z_h = z['y2'] - z['y1']
        z_w = z_h * MODEL_RATIO
        x1 = max(0, (img_w - z_w) / 2)
        x2 = min(img_w, x1 + z_w)
        bbox = np.array([x1, z['y1'], x2, z['y2']], dtype=np.float32)

        res = inference_topdown(model, img, bboxes=bbox[None, :])[0]
        kpts = res.pred_instances.keypoints[0]
        scores = res.pred_instances.keypoint_scores[0]

        # --- THE SANITY GUARD (Fix for Demo 7) ---
        # p0: Red (Pastern Top), p1: Orange (Pastern Bottom/Joint)
        # In a real horse, p0 MUST be significantly higher (smaller Y) than the hoof points.
        is_sane = True
        p0_y = kpts[0][1]
        p2_y = kpts[2][1] # Hoof Point
        
        # If the Pastern Top is lower than the Hoof Top, it's a "Heel Slip"
        if p0_y > p2_y - 10: # 10px buffer
            is_sane = False
            
        agg = np.mean(scores) * 10
        if not is_sane:
            agg -= 8  # Heavy penalty for anatomically impossible poses
            print(f"  - [{z['name']}]: REJECTED (Anatomy Sanity Check Failed)")
        else:
            print(f"  - [{z['name']}]: Score={agg:.2f} (Avg_Conf={np.mean(scores):.2f})")

        if agg > best_score or best_res is None:
            best_score = agg
            best_res = res
            best_zone = z['name']

    keypoints = best_res.pred_instances.keypoints[0]
    scores = best_res.pred_instances.keypoint_scores[0]

    vis = img.copy()

    colors = [(0,0,255),(0,165,255),(0,255,0),(255,0,0)]
    for i,(x,y) in enumerate(keypoints):
        cv2.circle(vis,(int(x),int(y)),6,colors[i],-1)
        cv2.putText(vis,f"{scores[i]:.2f}",(int(x)+4,int(y)-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,colors[i],1)

    if all(s > 0.10 for s in scores):
        # 1. Coordinate Prep (Deep copy to avoid reference leaks)
        pts_math = {i: np.array(keypoints[i], copy=True) for i in range(4)}
        pts_math = apply_anatomical_offset(pts_math, img_w)
        
        # 2. Math Normalization (Forward-Facing Invariant)
        # We transform the coordinates for math ONLY to handle left/right horses
        if pts_math[3][0] < pts_math[0][0]:
            cx = (pts_math[0][0] + pts_math[3][0]) / 2
            for i in pts_math:
                pts_math[i][0] = 2*cx - pts_math[i][0]

        # 3. Vector Calculation (Ensure vectors point UP for clinical angle)
        # Pastern: Bottom -> Top
        v_p = (pts_math[0][0] - pts_math[1][0], pts_math[0][1] - pts_math[1][1])
        # Hoof: Toe -> Top
        v_h = (pts_math[2][0] - pts_math[3][0], pts_math[2][1] - pts_math[3][1])

        p_angle = clinical_angle(angle_from_vertical(v_p))
        h_angle = clinical_angle(angle_from_vertical(v_h))
        diff = abs(p_angle - h_angle)

        # 4. FIXED DRAWING: Connect the dots (scale=1.0)
        p0, p1, p2, p3 = keypoints[0], keypoints[1], keypoints[2], keypoints[3]

        # Pastern Line: Connect p1 (Orange) to p0 (Red)
        draw_angle_line(vis, p1, p0, (0, 165, 255), scale=1.0) # Match Orange Marker

        # Hoof Line: Connect p3 (Blue) to p2 (Green)
        draw_angle_line(vis, p3, p2, (255, 128, 0), scale=1.0) # Match Blue Marker
        
        avg_conf = np.mean(scores)
        print(f"\n✅ HPA SUCCESS (Zone: {best_zone})")
        print(f"  Pastern: {p_angle:.1f} | Hoof: {h_angle:.1f} | Dev: {diff:.1f}")
        print(f"  Model Confidence: {avg_conf:.2f}")

        color = (0, 255, 0) if diff < 3 else (0, 0, 255)
        # Avoid ° symbol if it causes rendering issues
        cv2.putText(vis, f"HPA Dev: {diff:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(vis, f"Confidence: {avg_conf:.2f}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        print("\n❌ REJECTED: Low confidence")
        cv2.putText(vis,"REJECTED",(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imwrite(args.out_file, vis)
    print(f"Result saved to {args.out_file}")


if __name__ == "__main__":
    main()
