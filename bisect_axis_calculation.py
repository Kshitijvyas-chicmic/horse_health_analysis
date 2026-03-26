import cv2
import numpy as np
from sklearn.decomposition import PCA
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor
import math
import os

# -----------------------------
# Utility Functions
# -----------------------------

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

def axis_angle(axis):
    return np.degrees(np.arctan2(axis[1], axis[0]))

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    cos_theta = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle = np.arccos(cos_theta)
    angle_deg = np.degrees(angle)
    return 180 - angle_deg if angle_deg > 90 else angle_deg

# -----------------------------
# Unified Fitting Engine
# -----------------------------

def fit_clinical_axis(points_xy):
    """
    Unified fitting for both Pastern and Hoof.
    Points are (x, y). Fits x = m*y + b.
    Returns: axis_vector, anchor_center
    """
    pts = np.array(points_xy, dtype=np.float64)
    if len(pts) < 10: raise ValueError("Not enough points for fitting")
    
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Use random_state=42 for deterministic/reproducible results
    ransac = RANSACRegressor(residual_threshold=5.0, random_state=42)
    ransac.fit(ys.reshape(-1, 1), xs)
    
    m = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_
    
    # Vector points downward [dx, dy] where dy=1
    axis = np.array([m, 1.0])
    axis = axis / np.linalg.norm(axis)
    
    # Anchor to the geometric mean of pixels, projected onto the line
    c_y = np.mean(ys)
    c_x = m * c_y + b
    center = np.array([c_x, c_y])
    
    return axis, center

# -----------------------------
# Analysis Function
# -----------------------------

def analyze_leg(model_path, img_path, model_name):
    print(f"\n--- Analyzing with {model_name} Model ---")
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    results = model(img_path, verbose=False, deterministic=True)
    
    pastern_candidates = []
    hoof_candidates = []
    for r in results:
        if r.masks is None: continue
        masks_data = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for mask, cls in zip(masks_data, classes):
            mask = cv2.resize((mask > 0.5).astype(np.uint8), (w, h))
            if np.sum(mask) < 500: continue
            
            if int(cls) == 0: pastern_candidates.append(mask)
            elif int(cls) == 1: hoof_candidates.append(mask)

    if not pastern_candidates or not hoof_candidates:
        return {"error": "Missing pastern or hoof detection"}

    # 1. Selection Strategy
    pastern_mask = max(pastern_candidates, key=lambda m: np.sum(m))
    p_ys, p_xs = np.where(pastern_mask == 1)
    p_bottom = np.max(p_ys)
    p_center_x = np.mean(p_xs)
    
    def hoof_dist(h_mask):
        h_ys, h_xs = np.where(h_mask == 1)
        return abs(np.min(h_ys) - p_bottom) + abs(np.mean(h_xs) - p_center_x) * 1.5
    hoof_mask = min(hoof_candidates, key=hoof_dist)

    # Orientation
    h_ys, h_xs = np.where(hoof_mask == 1)
    hoof_center_x = np.mean(h_xs)
    orientation = 'left' if hoof_center_x < p_center_x else 'right'

    # 2. Pastern ROI Calculation (10-60%)
    p_y_min, p_y_max = np.min(p_ys), np.max(p_ys)
    p_y_range = p_y_max - p_y_min
    p_roi_top = p_y_min + p_y_range * 0.10
    p_roi_bottom = p_y_min + p_y_range * 0.60
    
    pastern_front_pts = []
    pastern_mid_pts = []
    unique_p_ys = np.unique(p_ys[(p_ys >= p_roi_top) & (p_ys <= p_roi_bottom)])
    for y_v in unique_p_ys:
        row_xs = p_xs[p_ys == y_v]
        x_min, x_max = np.min(row_xs), np.max(row_xs)
        # Front edge defines the bone-parallel slope
        pastern_front_pts.append([x_min if orientation == 'left' else x_max, y_v])
        # Midpoint defines the bisecting position
        pastern_mid_pts.append([(x_min + x_max) / 2, y_v])
    
    p_slope_axis, _ = fit_clinical_axis(pastern_front_pts)
    # Restore position to the middle of the leg
    p_center = np.mean(pastern_mid_pts, axis=0)
    p_axis = p_slope_axis
    
    # 3. Hoof ROI Calculation (20-75%) - Clinical Dorsal Wall Standard
    h_y_min, h_y_max = np.min(h_ys), np.max(h_ys)
    h_y_range = h_y_max - h_y_min
    h_roi_top = h_y_min + h_y_range * 0.20
    h_roi_bottom = h_y_min + h_y_range * 0.75
    
    hoof_pts = []
    unique_h_ys = np.unique(h_ys[(h_ys >= h_roi_top) & (h_ys <= h_roi_bottom)])
    for y_v in unique_h_ys:
        row_xs = h_xs[h_ys == y_v]
        hoof_pts.append([np.min(row_xs) if orientation == 'left' else np.max(row_xs), y_v])
    
    h_axis, h_center = fit_clinical_axis(hoof_pts)

    # Clinical Angles
    p_angle_c = clinical_angle(angle_from_vertical(p_axis))
    h_angle_c = clinical_angle(angle_from_vertical(h_axis))
    diff_c = abs(p_angle_c - h_angle_c)

    # 4. Visualization
    res_img = img.copy()
    
    # Draw Pastern ROI Points (Green dots - Front Edge)
    for pt in pastern_front_pts:
        cv2.circle(res_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    
    # Draw Hoof ROI Points (Yellow dots)
    for pt in hoof_pts:
        cv2.circle(res_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    scale = max(h, w) * 1.25
    
    # Pastern Line (Red)
    cv2.line(res_img, 
             (int(p_center[0] - p_axis[0]*scale), int(p_center[1] - p_axis[1]*scale)),
             (int(p_center[0] + p_axis[0]*scale), int(p_center[1] + p_axis[1]*scale)), 
             (0, 0, 255), 3)
    
    # Hoof Line (Yellow)
    cv2.line(res_img, 
             (int(h_center[0] - h_axis[0]*scale), int(h_center[1] - h_axis[1]*scale)),
             (int(h_center[0] + h_axis[0]*scale), int(h_center[1] + h_axis[1]*scale)), 
             (0, 255, 255), 3)
    
    # Overlays
    res_img[pastern_mask == 1] = res_img[pastern_mask == 1] * 0.7 + np.array([0, 0, 150]) * 0.3
    res_img[hoof_mask == 1] = res_img[hoof_mask == 1] * 0.7 + np.array([150, 0, 0]) * 0.3
    
    out_name = f"result_{model_name.lower()}.png"
    cv2.imwrite(out_name, res_img)
    
    return {
        "p_angle": p_angle_c,
        "h_angle": h_angle_c,
        "diff": diff_c,
        "img": out_name,
        "orientation": orientation
    }

# -----------------------------
# Execution
# -----------------------------

img_target = "/home/chetan/AI_First/horse_health_analysis/horse_health_analysis/round_hoof.png"
models = {
    "Medium": "runs/segment/hpa_v8m_full_v1/weights/best.pt",
    "Nano": "runs/segment/hpa_v8n_full_v1/weights/best.pt"
}

results_summary = {}
for name, path in models.items():
    try: results_summary[name] = analyze_leg(path, img_target, name)
    except Exception as e: results_summary[name] = {"error": str(e)}

print("\n" + "="*40)
print(f"COMPARISON FOR: {os.path.basename(img_target)}")
print("="*40)
print(f"{'Metric':<20} | {'Nano':<10} | {'Medium':<10}")
print("-" * 45)
metrics = [("p_angle", "Pastern Angle"), ("h_angle", "Hoof Angle"), ("diff", "HPA Difference")]
for key, label in metrics:
    v_n, v_m = results_summary['Nano'].get(key, "N/A"), results_summary['Medium'].get(key, "N/A")
    if isinstance(v_n, float): v_n = f"{v_n:.2f}°"
    if isinstance(v_m, float): v_m = f"{v_m:.2f}°"
    print(f"{label:<20} | {v_n:<10} | {v_m:<10}")
print("-" * 45)
print(f"{'Orientation':<20} | {results_summary['Nano'].get('orientation', 'N/A'):<10} | {results_summary['Medium'].get('orientation', 'N/A'):<10}")
print("="*40)
print(f"Results saved to: result_nano.png, result_medium.png")
