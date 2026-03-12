import cv2
import numpy as np
import os
import sys
import json
import math

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# --- HPA Calculation Logic (Pure Functions - Independently Testable) ---
def angle_from_vertical(v):
    """
    Calculate angle from vertical axis.
    Args:
        v: tuple (dx, dy) representing a vector
    Returns:
        float: angle in degrees from vertical
    """
    vertical = (0, -1)  # Y increases downward in images
    dot = v[0]*vertical[0] + v[1]*vertical[1]
    det = v[0]*vertical[1] - v[1]*vertical[0]
    return math.degrees(math.atan2(det, dot))

def clinical_angle(image_angle):
    """
    Convert image-based angle to clinical angle (Ground-relative).
    """
    angle = abs(image_angle) % 180
    if angle > 90:
        angle = 180 - angle
    return 90 - angle

def calculate_hpa_metrics(keypoints, normalize_direction=True):
    """
    Calculate HPA (Horse Pastern Axis) metrics from keypoints.
    
    This is a PURE FUNCTION - completely independent of visualization.
    
    Args:
        keypoints: dict or list with 4 keypoints
                  {0: (x,y), 1: (x,y), 2: (x,y), 3: (x,y)}
                  or [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
        normalize_direction: bool, if True, canonicalize leg direction
                            to ensure toe is always to the right of pastern top.
                            This is a safety invariant for physics-based angles.
    
    Returns:
        dict: {
            'pastern_angle': float,  # Anatomical angle (0-90°)
            'hoof_angle': float,     # Anatomical angle (0-90°)
            'deviation': float,      # Absolute difference
            'pastern_vector': tuple, # (dx, dy)
            'hoof_vector': tuple,    # (dx, dy)
            'was_flipped': bool      # True if direction was normalized
        }
    """
    # Convert to dict and handle as lists for mutability
    if isinstance(keypoints, (list, tuple)):
        pts = {i: list(keypoints[i]) for i in range(4)}
    else:
        pts = {i: list(keypoints[i]) for i in range(4)}
    
    was_flipped = False
    
    # === DIRECTION NORMALIZATION (Safety Invariant) ===
    # Goal: Decouple physical measurement from pose orientation.
    # We enforce that 'Forward' (toe) is always 'Positive X' (right).
    if normalize_direction:
        pastern_top_x = pts[0][0]
        toe_tip_x = pts[3][0]
        
        # If toe is to the left of pastern top, flip the coordinate system
        if toe_tip_x < pastern_top_x:
            # Shift center to midpoint of the segment to avoid large coordinate jumps
            center_x = (pastern_top_x + toe_tip_x) / 2
            for i in range(4):
                pts[i][0] = 2 * center_x - pts[i][0]
            was_flipped = True
    # ===================================================
    
    # Calculate vectors
    v_pastern = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    v_hoof = (pts[3][0] - pts[2][0], pts[3][1] - pts[2][1])
    
    # Calculate angles
    # Angles
    pastern_angle = clinical_angle(angle_from_vertical(v_pastern))
    hoof_angle = clinical_angle(angle_from_vertical(v_hoof))
    
    deviation = abs(pastern_angle - hoof_angle)
    
    return {
        'pastern_angle': pastern_angle,
        'hoof_angle': hoof_angle,
        'deviation': deviation,
        'pastern_vector': v_pastern,
        'hoof_vector': v_hoof,
        'was_flipped': was_flipped
    }
# -------------------------------------------------------------------

class DebugPoseVisualizer:
    # Keypoint colors for clear visibility
    KP_COLORS = {
        0: (0, 0, 255),     # pastern top = red
        1: (0, 165, 255),   # pastern bottom = orange
        2: (0, 255, 0),     # hoof wall top = green
        3: (255, 0, 0),     # toe tip = blue
    }

    KP_NAMES = {
        0: "pastern_top",
        1: "pastern_bottom",
        2: "hoof_wall_top",
        3: "toe_tip"
    }

    def __init__(self, config, checkpoint, device="cpu"):
        from mmengine.config import Config
        cfg = Config.fromfile(config)
        cfg.model.test_cfg.flip_test = True
        self.model = init_model(cfg, checkpoint, device=device)

    def run(
        self,
        image_path,
        json_path=None, # Add json path argument
        out_file="outputs/debug_pose.jpg",
        score_thr=0.0,
        radius=6
    ):
        print("Running debug visualizer (inside run method)...")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        h, w = img.shape[:2]

        # -------------------------------
        # Dynamic BBox Lookup
        # -------------------------------
        bbox = None
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Find image id by filename
                file_name = os.path.basename(image_path)
                image_id = None
                for img_entry in data['images']:
                     if img_entry['file_name'] == file_name:
                         image_id = img_entry['id']
                         break
                
                if image_id is not None:
                    # Find annotation for this image
                    for ann in data['annotations']:
                        if ann['image_id'] == image_id:
                            # COCO bbox: [x, y, w, h] -> XYXY: [x, y, x+w, y+h]
                            bx, by, bw, bh = ann['bbox']
                            bbox = np.array([[bx, by, bx+bw, by+bh]], dtype=np.float32)
                            print(f"✅ Found bbox in JSON for {file_name}: {bbox}")
                            break
            except Exception as e:
                print(f"⚠ Failed to load bbox from JSON: {e}")
        
        if bbox is None:
            print("⚠ No bbox found in JSON (or no JSON provided). Using full image (expect poor results).")
            bbox = np.array([[0, 0, w, h]], dtype=np.float32)

        # Inference returns a list of data sample objects
        results_list = inference_topdown(
            self.model,
            img,
            bbox,
            bbox_format="xyxy"
        )
        
        # FIX 1: Access the single data sample object from the returned list
        data_sample = results_list[0] 


        if data_sample.pred_instances.keypoints.size == 0:
            print("❌ No pose detected")
            return

        keypoints = data_sample.pred_instances.keypoints[0]    # (K, 2)
        scores = data_sample.pred_instances.keypoint_scores[0] # (K,)


        # Print predictions (Minimal print statements to avoid errors)
        print("\n🔍 Model predictions (Simplified Output):")
        for i, (pt, sc) in enumerate(zip(keypoints, scores)):
            # FIX 2: Explicitly unpack x and y coordinates and cast to int/float
            x, y = int(pt[0]), int(pt[1])
            score = float(sc)
            print(f"  KP {i}: x={x}, y={y}, score={score:.3f}")


        # Draw keypoints
        debug_img = img.copy()
        for i, (pt, sc) in enumerate(zip(keypoints, scores)):
            # FIX 3: Use the explicitly unpacked x and y coordinates
            x, y = int(pt[0]), int(pt[1])
            color = self.KP_COLORS[i] if sc >= score_thr else (0, 0, 0)
            if color != (0, 0, 0): # Only draw if visible (score above threshold)
                cv2.circle(debug_img, (x, y), radius, color, -1)
                cv2.putText(
                    debug_img,
                    f"{i}:{sc:.2f}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )

        # -------------------------------
        # HPA (Horse Pastern Axis) Analysis
        # -------------------------------
        # Map: 0:pastern_top, 1:pastern_bottom, 2:hoof_wall_top, 3:toe_tip
        if all(scores[i] >= score_thr for i in range(4)):
            pts = {i: (int(keypoints[i][0]), int(keypoints[i][1])) for i in range(4)}
            
            # === PURE CALCULATION (Independent of Visualization) ===
            # This function can be tested separately to verify correctness
            hpa_metrics = calculate_hpa_metrics(pts)
            
            angle1 = hpa_metrics['pastern_angle']
            angle2 = hpa_metrics['hoof_angle']
            diff = hpa_metrics['deviation']
            # ======================================================

            # Draw Lines (Visualization Only - Does NOT affect calculation)
            cv2.line(debug_img, pts[0], pts[1], (255, 128, 0), 2) # Orange (Pastern)
            cv2.line(debug_img, pts[2], pts[3], (0, 128, 255), 2) # Blue (Hoof)

            print(f"\n🐴 HPA Analysis (Validated Logic):")
            print(f"  Pastern Angle: {angle1:.1f}°")
            print(f"  Hoof Angle:    {angle2:.1f}°")
            print(f"  Deviation:     {diff:.1f}°")

            # Display metrics on image
            text_color = (0, 255, 0) if diff < 3 else (0, 0, 255)
            cv2.putText(debug_img, f"HPA Dev: {diff:.1f} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # Draw bounding box used for inference
        x1, y1, x2, y2 = bbox.flatten().astype(int) # Flatten bbox array before unpacking
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        cv2.imwrite(out_file, debug_img)
        print(f"\n✅ Debug image saved to: {out_file}")


# -------------------------------
# Example usage (Use your file paths here)
# -------------------------------

if __name__ == "__main__":
    # Ensure these paths are correct relative to where you run the script
    pose_config = "custom_configs/rtmpose_hoof_4kp.py"
    pose_checkpoint = "work_dirs/rtmpose_hoof_4kp_portrait/epoch_100.pth"
    image_path = "/home/chetan/AI_First/horse_health_analysis/horse_health_analysis/data/images/val/h138.jpeg"

    try:
        print("Attempting to initialize model...")
        debugger = DebugPoseVisualizer(pose_config, pose_checkpoint, device="cuda:0")
        print("Model initialized successfully. Running visualizer...")
        debugger.run(
            image_path,
            json_path="../data/annotations/val_fixed.json",
            out_file="outputs/hoof_debug_full_image_bbox.jpg",
            score_thr=0.0
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found. Check your paths.")
        print(f"Missing file: {e.filename}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during initialization or runtime:")
        import traceback; traceback.print_exc()
