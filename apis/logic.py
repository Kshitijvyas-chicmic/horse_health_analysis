import cv2
import numpy as np
import math
import os
import base64
import sys
sys.path.append('mmpose')
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmengine.config import Config

# --- ANGLE MATH ---
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

def apply_anatomical_offset(pts, img_w):
    is_left_facing = pts[2][0] < pts[1][0]
    direction = -1 if is_left_facing else 1
    width_estimate = img_w * 0.05 
    pts[0][0] += direction * (width_estimate * 0.25)
    pts[1][0] += direction * (width_estimate * 0.10)
    return pts

def draw_angle_line(img, p_start, p_end, color, scale=1.5, thickness=3):
    p_start = np.array(p_start, dtype=np.float32)
    p_end = np.array(p_end, dtype=np.float32)
    v = p_end - p_start
    length = np.linalg.norm(v)
    if length < 1e-3: return
    v = v / length
    line_len = length * scale
    p_draw_end = p_start + v * line_len
    cv2.line(img, tuple(p_start.astype(int)), tuple(p_draw_end.astype(int)), color, thickness, cv2.LINE_AA)

class HPAPredictor:
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        register_all_modules()
        cfg = Config.fromfile(config_path)
        
        # flip_test: Runs inference twice (normal + flipped) and averages results
        # - Accuracy gain: ~0.5-2% improvement
        # - Speed cost: 2x slower (doubles inference time)
        # - Current setting: False (prioritizing speed for production)
        # - To re-enable: Set to True (recommended if using Gunicorn with 4+ workers)
        cfg.model.test_cfg.flip_test = False
        
        self.model = init_model(cfg, checkpoint_path, device=device)
        self.MODEL_RATIO = 192 / 256
        
    def predict(self, img_bytes):
        # Prevent empty or None buffers
        if not img_bytes:
            raise ValueError("Empty image buffer provided")
            
        # Convert bytes to cv2 image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
            
        img_h, img_w = img.shape[:2]
        
        zones = [
            {'name': 'Floor-Scan',   'y1': int(img_h * 0.4), 'y2': img_h},
            {'name': 'Anatomy-Scan', 'y1': int(img_h * 0.2), 'y2': int(img_h * 0.8)},
            {'name': 'Top-Anatomy',  'y1': 0,               'y2': int(img_h * 0.6)},
            {'name': 'Global-Scan',  'y1': 0,               'y2': img_h},
        ]
        
        best_res = None
        best_score = -1
        best_zone = None
        
        for z in zones:
            z_h = z['y2'] - z['y1']
            z_w = z_h * self.MODEL_RATIO
            x1 = max(0, (img_w - z_w) / 2)
            x2 = min(img_w, x1 + z_w)
            bbox = np.array([x1, z['y1'], x2, z['y2']], dtype=np.float32)
            
            res = inference_topdown(self.model, img, bboxes=bbox[None, :])[0]
            kpts = res.pred_instances.keypoints[0]
            scores = res.pred_instances.keypoint_scores[0]
            
            is_sane = True
            p0_y = kpts[0][1]
            p2_y = kpts[2][1]
            if p0_y > p2_y - 10:
                is_sane = False
            
            agg = np.mean(scores) * 10
            if not is_sane:
                agg -= 8
                
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
            
        metrics = {
            "success": False,
            "best_zone": best_zone,
            "pastern_angle": None,
            "hoof_angle": None,
            "hpa_dev": None,
            "image_base64": None
        }
        
        if all(s > 0.10 for s in scores):
            pts_math = {i: np.array(keypoints[i], copy=True) for i in range(4)}
            pts_math = apply_anatomical_offset(pts_math, img_w)
            
            if pts_math[3][0] < pts_math[0][0]:
                cx = (pts_math[0][0] + pts_math[3][0]) / 2
                for i in pts_math:
                    pts_math[i][0] = 2*cx - pts_math[i][0]
            
            v_p = (pts_math[0][0] - pts_math[1][0], pts_math[0][1] - pts_math[1][1])
            v_h = (pts_math[2][0] - pts_math[3][0], pts_math[2][1] - pts_math[3][1])
            
            p_angle = clinical_angle(angle_from_vertical(v_p))
            h_angle = clinical_angle(angle_from_vertical(v_h))
            diff = abs(p_angle - h_angle)
            
            p0, p1, p2, p3 = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
            draw_angle_line(vis, p1, p0, (0, 165, 255), scale=1.0)
            draw_angle_line(vis, p3, p2, (255, 128, 0), scale=1.0)
            
            color = (0, 255, 0) if diff < 3 else (0, 0, 255)
            cv2.putText(vis, f"HPA Dev: {diff:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            metrics.update({
                "success": True,
                "pastern_angle": round(p_angle, 2),
                "hoof_angle": round(h_angle, 2),
                "hpa_dev": round(diff, 2),
                "model_confidence": round(float(np.mean(scores)), 2)
            })
        else:
            cv2.putText(vis,"REJECTED",(20,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
        _, buffer = cv2.imencode('.jpg', vis)
        metrics["image_base64"] = base64.b64encode(buffer).decode('utf-8')
        
        return metrics
