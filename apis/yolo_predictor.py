"""
YOLOPredictor — Production YOLO Medium segmentation predictor.

This class mirrors the interface of HPAPredictor (logic.py) so that
both models can be used interchangeably in the API layer.

predict(image_bytes: bytes) -> dict
  Returns:
    success            bool
    pastern_angle      float | None
    hoof_angle         float | None
    hpa_dev            float | None
    model_confidence   float | None  (average detection confidence 0-1)
    error              str  | None
    image_base64       str  | None

IMPORTANT: This module is not model-training code. It is purely
an inference/prediction class for the API server.
"""

import cv2
import numpy as np
import base64
import math
import threading
import os

from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor


# ─────────────────────────────────────────────
# Math helpers (same as bisect_axis_calculation.py)
# ─────────────────────────────────────────────

def _angle_from_vertical(v):
    """Returns signed angle (degrees) between vector v and (0, -1) [straight up]."""
    vertical = (0, -1)
    dot = v[0] * vertical[0] + v[1] * vertical[1]
    det = v[0] * vertical[1] - v[1] * vertical[0]
    return math.degrees(math.atan2(det, dot))


def _clinical_angle(image_angle):
    """Converts raw signed image angle to always-positive clinical angle."""
    angle = abs(image_angle) % 180
    if angle > 90:
        angle = 180 - angle
    return 90 - angle


def _fit_clinical_axis(points_xy):
    """
    Unified RANSAC slope fitting (identical to bisect_axis_calculation).
    Fits x = m*y + b. Returns (axis_vector, anchor_center).
    Uses random_state=42 for deterministic results.
    """
    pts = np.array(points_xy, dtype=np.float64)
    if len(pts) < 10:
        raise ValueError("Not enough ROI points for RANSAC fit")

    xs, ys = pts[:, 0], pts[:, 1]
    ransac = RANSACRegressor(residual_threshold=5.0, random_state=42)
    ransac.fit(ys.reshape(-1, 1), xs)

    m = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_

    axis = np.array([m, 1.0])
    axis /= np.linalg.norm(axis)

    c_y = np.mean(ys)
    center = np.array([m * c_y + b, c_y])

    return axis, center


# ─────────────────────────────────────────────
# YOLOPredictor class
# ─────────────────────────────────────────────

class YOLOPredictor:
    """
    Production YOLO Medium predictor for the Horse Health API.
    Thread-safe: each predict() call creates its own local state.
    Model is loaded once at server startup via __init__.
    """

    # Class-level lock to avoid parallel YOLO model init races
    _load_lock = threading.Lock()

    def __init__(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        with self._load_lock:
            self.model = YOLO(weights_path)

        # Warm-up: run a tiny dummy inference to allocate GPU/CPU memory
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

        self.weights_path = weights_path
        print(f"✅ YOLOPredictor loaded: {weights_path}")

    def predict(self, img_bytes: bytes) -> dict:
        """
        Runs hoof+pastern segmentation and computes clinical angles.

        Returns dict with keys:
            success, pastern_angle, hoof_angle, hpa_dev,
            model_confidence, error, image_base64
        """
        metrics = {
            "success": False,
            "pastern_angle": None,
            "hoof_angle": None,
            "hpa_dev": None,
            "model_confidence": None,
            "error": None,
            "image_base64": None,
        }

        if not img_bytes:
            metrics["error"] = "Empty image buffer"
            return metrics

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            metrics["error"] = "Could not decode image"
            return metrics

        h, w = img.shape[:2]

        try:
            results = self.model(img, verbose=False, deterministic=True)
        except Exception as e:
            metrics["error"] = f"YOLO inference error: {e}"
            return metrics

        pastern_candidates = []  # (mask, conf)
        hoof_candidates = []     # (mask, conf)

        for r in results:
            if r.masks is None:
                continue
            masks_data = r.masks.data.cpu().numpy()
            classes    = r.boxes.cls.cpu().numpy()
            confs      = r.boxes.conf.cpu().numpy()   # ← detection confidence

            for mask, cls, conf in zip(masks_data, classes, confs):
                mask_bin = cv2.resize((mask > 0.5).astype(np.uint8), (w, h))
                if np.sum(mask_bin) < 500:
                    continue
                if int(cls) == 0:
                    pastern_candidates.append((mask_bin, float(conf)))
                elif int(cls) == 1:
                    hoof_candidates.append((mask_bin, float(conf)))

        if not pastern_candidates or not hoof_candidates:
            metrics["error"] = "Could not detect both pastern and hoof. Poor image quality or incorrect angle. Please retake the image."
            return metrics

        # ── Selection: largest-area mask wins ──
        pastern_mask, p_conf = max(pastern_candidates, key=lambda x: np.sum(x[0]))
        p_ys, p_xs = np.where(pastern_mask == 1)
        p_bottom   = np.max(p_ys)
        p_center_x = np.mean(p_xs)

        def _hoof_dist(hm_pair):
            hm = hm_pair[0]
            h_ys, h_xs = np.where(hm == 1)
            return abs(np.min(h_ys) - p_bottom) + abs(np.mean(h_xs) - p_center_x) * 1.5

        hoof_mask, h_conf = min(hoof_candidates, key=_hoof_dist)

        # Overall model confidence = average of the two selected detections
        model_confidence = round((p_conf + h_conf) / 2.0, 4)
        metrics["model_confidence"] = model_confidence

        # ── Orientation ──
        h_ys, h_xs = np.where(hoof_mask == 1)
        orientation = "left" if np.mean(h_xs) < p_center_x else "right"

        try:
            # ── Pastern ROI (10-60%) ──
            p_y_min, p_y_max = np.min(p_ys), np.max(p_ys)
            p_y_range = p_y_max - p_y_min
            p_roi_top    = p_y_min + p_y_range * 0.10
            p_roi_bottom = p_y_min + p_y_range * 0.60

            pastern_front_pts = []
            pastern_mid_pts   = []
            for y_v in np.unique(p_ys[(p_ys >= p_roi_top) & (p_ys <= p_roi_bottom)]):
                row_xs = p_xs[p_ys == y_v]
                x_min, x_max = np.min(row_xs), np.max(row_xs)
                pastern_front_pts.append([x_min if orientation == "left" else x_max, y_v])
                pastern_mid_pts.append([(x_min + x_max) / 2, y_v])

            p_slope_axis, _ = _fit_clinical_axis(pastern_front_pts)
            p_center = np.mean(pastern_mid_pts, axis=0)

            # ── Hoof ROI (20-75%) ──
            h_y_min, h_y_max = np.min(h_ys), np.max(h_ys)
            h_y_range = h_y_max - h_y_min
            h_roi_top    = h_y_min + h_y_range * 0.20
            h_roi_bottom = h_y_min + h_y_range * 0.75

            hoof_pts = []
            for y_v in np.unique(h_ys[(h_ys >= h_roi_top) & (h_ys <= h_roi_bottom)]):
                row_xs = h_xs[h_ys == y_v]
                hoof_pts.append([np.min(row_xs) if orientation == "left" else np.max(row_xs), y_v])

            h_axis, h_center = _fit_clinical_axis(hoof_pts)

        except ValueError as e:
            metrics["error"] = f"ROI fitting error: {e}"
            return metrics

        # ── Clinical Angles ──
        p_angle = round(_clinical_angle(_angle_from_vertical(p_slope_axis)), 2)
        h_angle = round(_clinical_angle(_angle_from_vertical(h_axis)), 2)
        hpa_dev = round(abs(p_angle - h_angle), 2)

        metrics.update({
            "success": True,
            "pastern_angle": p_angle,
            "hoof_angle": h_angle,
            "hpa_dev": hpa_dev,
        })

        # ── Visualization ──
        vis = img.copy()
        vis[pastern_mask == 1] = vis[pastern_mask == 1] * 0.7 + np.array([0, 0, 150]) * 0.3
        vis[hoof_mask == 1]    = vis[hoof_mask == 1]    * 0.7 + np.array([150, 0, 0]) * 0.3

        scale = max(h, w) * 1.25
        cv2.line(vis,
                 (int(p_center[0] - p_slope_axis[0]*scale), int(p_center[1] - p_slope_axis[1]*scale)),
                 (int(p_center[0] + p_slope_axis[0]*scale), int(p_center[1] + p_slope_axis[1]*scale)),
                 (0, 0, 255), 3)
        cv2.line(vis,
                 (int(h_center[0] - h_axis[0]*scale), int(h_center[1] - h_axis[1]*scale)),
                 (int(h_center[0] + h_axis[0]*scale), int(h_center[1] + h_axis[1]*scale)),
                 (0, 255, 255), 3)

        _, buf = cv2.imencode(".jpg", vis)
        metrics["image_base64"] = base64.b64encode(buf).decode("utf-8")

        return metrics
