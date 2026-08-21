
from pathlib import Path
import glob
import argparse
import logging
import cv2
import numpy as np
import sys
from PIL import Image as PILImage
from mmpose.apis import MMPoseInferencer
import json
import base64

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_depth_pipe = None

def get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device_id = 0 if torch.cuda.is_available() else -1
        logging.info(f"Loading Depth Anything V2 Small on device {device_id} ...")
        _depth_pipe = hf_pipeline("depth-estimation", model=_DEPTH_MODEL_ID, device=device_id)
    return _depth_pipe

def estimate_depth(image_bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    pipe = get_depth_pipe()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    out = pipe(pil_img)
    depth_map = np.array(out["depth"]).astype(np.float32)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    if fg_mask is not None:
        depth_map[fg_mask == 0] = 0.0
    return depth_map

# ---------------------------------------------------------------------------
# Foreground mask extraction from pre-removed background image
# ---------------------------------------------------------------------------

def extract_mask_from_processed(processed_path: str) -> tuple:
    """Load a background-removed image (transparent PNG) and extract its mask.

    Reads the alpha channel of the processed image and derives a clean binary
    foreground mask via thresholding + morphological cleanup.

    Returns:
        processed_bgra — the processed image as BGRA uint8 numpy array (H x W x 4)
        mask           — binary uint8 mask (255 = foreground, 0 = background)
    """
    processed_bgra = cv2.imread(processed_path, cv2.IMREAD_UNCHANGED)
    if processed_bgra is None:
        raise FileNotFoundError(f"Cannot read background-removed image: {processed_path}")
    if processed_bgra.ndim < 3 or processed_bgra.shape[2] < 4:
        raise ValueError(
            f"Expected a 4-channel (BGRA) image for the processed file, got shape {processed_bgra.shape}: {processed_path}"
        )
    alpha = processed_bgra[:, :, 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    logging.info("Extracted mask from processed image (%d fg pixels).", int(cv2.countNonZero(mask)))
    return processed_bgra, mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Legacy split_mask_on_width removed (Gemini handles touching objects)
# ---------------------------------------------------------------------------





def trim_upper_leg_fraction(leg_mask: np.ndarray, exclude_top_frac: float = 0.05) -> np.ndarray:
    """Remove the top fraction of a selected leg mask before analysis."""
    if leg_mask is None or leg_mask.size == 0:
        return leg_mask

    ys = np.where(np.any(leg_mask > 0, axis=1))[0]
    if ys.size == 0:
        return leg_mask.copy()

    top_y, bottom_y = int(ys[0]), int(ys[-1])
    leg_h = bottom_y - top_y + 1
    cut_y = top_y + int(round(leg_h * exclude_top_frac))
    cut_y = min(max(top_y, cut_y), bottom_y + 1)

    trimmed = leg_mask.copy()
    trimmed[top_y:cut_y, :] = 0
    return trimmed


# ---------------------------------------------------------------------------
# Legacy tail rejection removed (Gemini handles tail isolation)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Leg selection — fallback (no MMPose)
# ---------------------------------------------------------------------------

def select_front_leg_fallback(mask: np.ndarray,
                               debug: bool = False) -> np.ndarray | None:
    """Select the frontmost front leg using simple heuristics to generate a candidate blob.
    """
    h, w = mask.shape
    zone = np.zeros_like(mask)
    zone[int(h * 0.20):] = mask[int(h * 0.20):]

    raw_contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not raw_contours:
        return None

    img_cx = w / 2.0
    candidates = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx_part = bx + bw / 2.0
        dx = abs(cx_part - img_cx)

        part = np.zeros_like(mask)
        cv2.drawContours(part, [cnt], -1, 255, cv2.FILLED)

        bottom_width = 0
        for ry in range(by + int(bh * 0.75), min(by + bh + 1, h)):
            xs = np.where(part[ry] > 0)[0]
            if xs.size > 0:
                bottom_width = max(bottom_width, int(xs[-1] - xs[0]))

        if bottom_width < max(20, int(0.12 * w)):
            continue

        if debug:
            logging.info("Candidate: area=%.0f bottom_w=%d dx=%.1f",
                         area, bottom_width, dx)
        candidates.append({
            'mask': part, 'area': area,
            'bottom_width': bottom_width, 'dx': dx,
        })

    if not candidates:
        return None

    max_area = max(c['area'] for c in candidates)
    valid_candidates = [c for c in candidates if c['area'] >= max_area * 0.15]
    valid_candidates.sort(key=lambda c: c['area'], reverse=True)
    best = valid_candidates[0]
    logging.info("Selected front leg candidate blob (area=%.0f) for Gemini", best['area'])
    return best['mask']


# ---------------------------------------------------------------------------
# Leg selection — AI path (MMPose keypoints)
# ---------------------------------------------------------------------------

def select_front_leg_from_keypoints(mask: np.ndarray,
                                    knee: tuple[float, float],
                                    hoof: tuple[float, float],
                                    debug: bool = False) -> np.ndarray | None:
    """Select a leg contour that best matches provided knee/hoof keypoints.

    Receives depth-filtered mask (FIX #9).
    """
    h, w = mask.shape
    start_y = max(0, int(round(knee[1])) - 40)
    zone = np.zeros_like(mask)
    zone[start_y:] = mask[start_y:]
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hoof_pt = (float(hoof[0]), float(hoof[1]))
    best_cnt, best_score = None, None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        dist = cv2.pointPolygonTest(cnt, hoof_pt, True)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
        else:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = bx + bw // 2
        dx_knee = abs(cx - int(round(knee[0])))
        score = (dist, -dx_knee, area)
        if debug:
            logging.info("candidate: area=%d dist=%.2f dx_knee=%d", area, dist, int(dx_knee))
        if best_score is None or score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return None

    lm = np.zeros_like(mask)
    cv2.drawContours(lm, [best_cnt], -1, 255, cv2.FILLED)

    ky, hy = int(round(knee[1])), int(round(hoof[1]))
    top_clip = max(0, ky - 20)
    bottom_clip = min(mask.shape[0] - 1, hy + 60)
    band = np.zeros_like(mask)
    band[top_clip: bottom_clip + 1, :] = 1
    lm = cv2.bitwise_and(lm, lm, mask=band.astype(np.uint8))
    lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    lm = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    xs = np.where(lm[min(bottom_clip, mask.shape[0] - 1)] > 0)[0]
    if xs.size == 0:
        if debug:
            logging.info("rejecting candidate: no pixels at hoof row")
        return None
    if xs[-1] - xs[0] < max(20, int(0.08 * mask.shape[1])):
        if debug:
            logging.info("rejecting candidate: bottom too narrow")
        return None
    return lm


# ---------------------------------------------------------------------------
# Legacy watershed logic removed (Gemini handles touching hooves/legs)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MMPose helpers
# ---------------------------------------------------------------------------

def get_ai_leg_keypoints(inferencer,
                         image_path: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if inferencer is None:
        return []
    try:
        res = inferencer(image_path)
    except Exception as e:
        logging.warning("MMPose inference failed: %s", e)
        return []

    try:
        results = next(iter(res)) if (hasattr(res, '__iter__') and not isinstance(res, dict)) else res
    except Exception:
        results = res

    preds = None
    if isinstance(results, dict) and results.get('predictions'):
        preds = results['predictions'][0]
    elif isinstance(results, list) and results:
        preds = results[0]
    if not preds:
        return []

    kpts, scores = None, None
    if isinstance(preds, dict):
        kpts = preds.get('keypoints') or preds.get('preds')
        scores = preds.get('keypoint_scores') or preds.get('scores')
    if kpts is None:
        return []

    legs = []
    try:
        if isinstance(kpts, np.ndarray):
            kpts = kpts.tolist()
        if len(kpts) > 10:
            def sc(i):
                return float(scores[i]) if scores and len(scores) > i else 1.0
            if sc(6) > 0.12 and sc(7) > 0.12:
                legs.append((tuple(kpts[6][:2]), tuple(kpts[7][:2])))
            if sc(9) > 0.12 and sc(10) > 0.12:
                legs.append((tuple(kpts[9][:2]), tuple(kpts[10][:2])))
    except Exception:
        return []
    return legs


# ---------------------------------------------------------------------------
# FIX #8 (IMPLEMENTED) — Cannon-bone axis: strictly vertical centre line
# ---------------------------------------------------------------------------

def find_cannon_bone_axis(leg_mask: np.ndarray,
                          target_knee: tuple[float, float] | None = None,
                          target_hoof: tuple[float, float] | None = None
                          ) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return a strictly vertical centre-line axis for the cannon bone.

    FIX #8 (IMPLEMENTED): Uses the median X of cannon-zone row midpoints.
    Both pt_top and pt_bottom share the same X coordinate so the rendered
    line is always at 90° from the ground — no diagonal slant.

    FIX #10: Bottom of axis clamped to the last row with foreground pixels
    (bottom_y), not bottom_y + 120, which pushed the line below the hoof.
    """
    h, w = leg_mask.shape
    clean = cv2.morphologyEx(leg_mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ys_mask = np.where(np.any(clean > 0, axis=1))[0]
    if ys_mask.size == 0:
        return (w // 2, 0), (w // 2, h - 1)
    top_y, bottom_y = int(ys_mask[0]), int(ys_mask[-1])

    rows = []
    for ry in range(top_y, bottom_y + 1):
        xs = np.where(clean[ry] > 0)[0]
        if xs.size >= 2:
            rows.append((ry, int(xs[0]), int(xs[-1]),
                         (float(xs[0]) + float(xs[-1])) / 2.0,
                         int(xs[-1] - xs[0])))

    if not rows:
        return (w // 2, top_y), (w // 2, bottom_y)

    # Define cannon zone (10%–40% of leg height, or AI-guided)
    if target_knee is not None and target_hoof is not None:
        try:
            lh = float(target_hoof[1] - target_knee[1])
            cs = float(target_knee[1]) + lh * 0.10
            ce = float(target_knee[1]) + lh * 0.40
            cannon_rows = [r for r in rows if cs <= r[0] <= ce]
        except Exception:
            cannon_rows = []
    else:
        lh = bottom_y - top_y + 1
        cs = top_y + lh * 0.10
        ce = top_y + lh * 0.40
        cannon_rows = [r for r in rows if cs <= r[0] <= ce]

    if not cannon_rows:
        cannon_rows = rows

    fit_xs = np.array([r[3] for r in cannon_rows], dtype=np.float64)

    # FIX #8: use median X — gives a strict vertical line, no diagonal slope
    median_cx = int(round(np.median(fit_xs)))

    # FIX #10: clamp bottom to actual mask extent, not +120 px past it
    pt_top = (median_cx, top_y)
    pt_bottom = (median_cx, bottom_y)

    logging.info("Cannon axis (vertical): top=%s bottom=%s (median_cx=%d)",
                 pt_top, pt_bottom, median_cx)
    return pt_top, pt_bottom


# ---------------------------------------------------------------------------
# Symmetry analysis with vertical centre line
# ---------------------------------------------------------------------------

def analyze_symmetry(leg_mask: np.ndarray,
                     pt_top: tuple[int, int],
                     pt_bottom: tuple[int, int]):
    """Row-by-row symmetry analysis.

    With the vertical line fix (#8), cx_at(ry) always returns the same X
    (pt_top[0] == pt_bottom[0]), making split straightforward and accurate.
    """
    h, w = leg_mask.shape
    green = np.zeros((h, w), dtype=np.uint8)
    red = np.zeros((h, w), dtype=np.uint8)

    top_y, bottom_y = pt_top[1], pt_bottom[1]
    dy = bottom_y - top_y

    def cx_at(ry: int) -> int:
        """Centre X at row ry — constant for vertical line."""
        if dy == 0:
            return pt_top[0]
        t = (ry - top_y) / dy
        return int(round(pt_top[0] + t * (pt_bottom[0] - pt_top[0])))

    total_left = total_right = 0
    row_data = []
    for ry in range(top_y, min(bottom_y + 1, h)):
        xs = np.where(leg_mask[ry] > 0)[0]
        if xs.size < 2:
            row_data.append(None)
            continue
        lx, rx = int(xs[0]), int(xs[-1])
        cx = cx_at(ry)
        if lx >= cx or rx <= cx:
            row_data.append(None)
            continue
        lw = cx - lx
        rw = rx - cx
        total_left += lw
        total_right += rw
        row_data.append((ry, lx, rx, lw, rw, cx))

    if total_left > total_right * 1.02:
        dominant = "LEFT"
    elif total_right > total_left * 1.02:
        dominant = "RIGHT"
    else:
        dominant = "SYMMETRIC"

    for item in row_data:
        if item is None:
            continue
        ry, lx, rx, lw, rw, cx = item
        sw = min(lw, rw)
        if dominant == "LEFT":
            green[ry, cx: rx + 1] = leg_mask[ry, cx: rx + 1]
            green[ry, max(0, cx - sw): cx] = leg_mask[ry, max(0, cx - sw): cx]
            if lw > sw:
                es, ee = lx, max(0, cx - sw)
                if es < ee:
                    red[ry, es:ee] = leg_mask[ry, es:ee]
        elif dominant == "RIGHT":
            green[ry, lx: cx] = leg_mask[ry, lx: cx]
            green[ry, cx: min(w, cx + sw + 1)] = leg_mask[ry, cx: min(w, cx + sw + 1)]
            if rw > sw:
                es, ee = min(w, cx + sw + 1), rx + 1
                if es < ee:
                    red[ry, es:ee] = leg_mask[ry, es:ee]
        else:
            green[ry, lx: rx + 1] = leg_mask[ry, lx: rx + 1]

    green = cv2.bitwise_and(green, leg_mask)
    red = cv2.bitwise_and(red, leg_mask)
    logging.info("Dominant side: %s (left=%d right=%d)", dominant, total_left, total_right)
    return green, red, dominant


def apply_overlay(img: np.ndarray, green_mask: np.ndarray,
                  red_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    res = img.astype(np.float32)
    orig = res.copy()
    COLOR_GREEN = np.array([34, 197, 94], dtype=np.float32)
    COLOR_RED = np.array([48, 48, 220], dtype=np.float32)
    gm, rm = green_mask > 0, red_mask > 0
    res[gm] = orig[gm] * (1 - alpha) + COLOR_GREEN * alpha
    res[rm] = orig[rm] * (1 - alpha) + COLOR_RED * alpha
    return np.clip(res, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_image(original_path: str, processed_path: str,
                  do_debug: bool = False, inferencer=None, gemini_key: str = None) -> None:
    """Analyse horse leg symmetry.

    Parameters
    ----------
    original_path : str
        Path to the original (with background) image.  The final annotated
        image is rendered on top of this so the background is preserved.
    processed_path : str
        Path to the background-removed version of the same image (BGRA PNG
        with a transparent background).  Used as a stencil: the alpha channel
        is thresholded to obtain the foreground mask, and all analysis
        (depth estimation, leg isolation, symmetry) runs on this mask.
        The resulting red/green overlay is then stamped onto the original.
    """
    p = Path(original_path)
    p_processed = Path(processed_path)

    # --- Load original image (used only for the final rendered output) ---
    img = cv2.imread(str(p))
    if img is None:
        logging.error("Cannot read original image: %s", p)
        return
    h, w = img.shape[:2]
    logging.info("Processing %s (%dx%d)  |  processed: %s", p.name, w, h, p_processed.name)

    # --- Extract foreground mask from the pre-removed background image ---
    try:
        processed_bgra, mask = extract_mask_from_processed(str(p_processed))
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return

    # Resize processed mask to match original if dimensions differ
    if processed_bgra.shape[:2] != (h, w):
        logging.warning(
            "processed image size %s differs from original %s — resizing mask.",
            processed_bgra.shape[:2], (h, w)
        )
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        processed_bgra = cv2.resize(processed_bgra, (w, h), interpolation=cv2.INTER_LINEAR)

    logging.info("Estimating depth map for Gemini processing...")
    try:
        fg_bgr = processed_bgra[:, :, :3].copy()
        fg_bgr[mask == 0] = 127
        depth_map = estimate_depth(fg_bgr, mask)
        if do_debug:
            depth_8u = (depth_map * 255).astype(np.uint8)
            depth_heatmap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
            cv2.imwrite(str(p.parent / f"{p.stem}_depth.png"), depth_heatmap)
    except Exception as e:
        logging.warning("Depth estimation failed: %s", e)
        depth_map = None
    depth_mask = mask.copy()

    leg_masks = []
    leg_infos = []

    # --- AI path (MMPose keypoints on the original image) ---
    if inferencer is not None:
        try:
            legs = get_ai_leg_keypoints(inferencer, str(p))
        except Exception:
            legs = []
        if legs:
            # Pick the leg closest to the center line
            best_leg = None
            min_dx = float('inf')
            img_cx = w / 2.0
            for knee, hoof in legs:
                cx = (knee[0] + hoof[0]) / 2.0
                dx = abs(cx - img_cx)
                if dx < min_dx:
                    min_dx = dx
                    best_leg = (knee, hoof)

            if best_leg is not None:
                knee, hoof = best_leg
                lm = select_front_leg_from_keypoints(depth_mask, knee, hoof, debug=do_debug)
                if lm is not None:
                    lm = trim_upper_leg_fraction(lm, 0.05)
                    leg_masks.append(lm)
                    leg_infos.append({'mask': lm, 'knee': knee, 'hoof': hoof})
                    cv2.imwrite(str(p.parent / f"{p.stem}_isolated_leg.png"), lm)
                    logging.info("Saved isolated leg (AI path).")

    # --- Fallback path ---
    if not leg_masks:
        logging.warning("AI keypoints missing or failed — using fallback leg selection.")
        lm = select_front_leg_fallback(depth_mask, debug=do_debug)
        if lm is None:
            logging.warning("No front leg found for %s", p.name)
            cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), img)
            return
        lm = trim_upper_leg_fraction(lm, 0.20)
        leg_masks = [lm]
        leg_infos = [{'mask': lm, 'knee': (w / 2.0, 0.0), 'hoof': (w / 2.0, float(h - 1))}]
        cv2.imwrite(str(p.parent / f"{p.stem}_isolated_leg.png"), lm)
        logging.info("Saved isolated leg (fallback path).")

    # --- Symmetry analysis (masks computed from nobg) ---
    combined_green = np.zeros((h, w), dtype=np.uint8)
    combined_red = np.zeros((h, w), dtype=np.uint8)
    per_leg_draw: list[dict] = []

    def refine_mask_with_gemini(original_bgr: np.ndarray, mask: np.ndarray, api_key: str, nobg_bgra: np.ndarray = None, depth_map: np.ndarray = None) -> np.ndarray:
        if genai is None:
            logging.warning("google-generativeai is not installed. Cannot use Gemini.")
            return mask

        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return mask
        
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        pad = 20
        hm, wm = mask.shape
        c_y_min, c_y_max = max(0, y_min - pad), min(hm, y_max + pad)
        c_x_min, c_x_max = max(0, x_min - pad), min(wm, x_max + pad)

        # Prefer the background-removed image so Gemini sees only the clean horse
        # silhouette (no barn, ground, other horses). This makes leg/tail distinction easier.
        if nobg_bgra is not None:
            crop_bgr = nobg_bgra[c_y_min:c_y_max, c_x_min:c_x_max, :3].copy()
            # Set pixels with alpha below threshold to white so the silhouette reads clearly on a neutral bg
            alpha = nobg_bgra[c_y_min:c_y_max, c_x_min:c_x_max, 3]
            crop_bgr[alpha < 128] = 255
        else:
            crop_bgr = original_bgr[c_y_min:c_y_max, c_x_min:c_x_max]
        
        ch, cw = crop_bgr.shape[:2]
        crop_mask = mask[c_y_min:c_y_min+ch, c_x_min:c_x_min+cw].copy()

        _, buffer = cv2.imencode('.jpg', crop_bgr)
        b64_str = base64.b64encode(buffer).decode('utf-8')

        depth_b64_str = None
        depth_color = None
        if depth_map is not None:
            crop_depth = depth_map[c_y_min:c_y_max, c_x_min:c_x_max]
            depth_8u = (crop_depth * 255).astype(np.uint8)
            # Create a colored heatmap for Gemini (TURBO colormap)
            depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
            _, dbuf = cv2.imencode('.jpg', depth_color)
            depth_b64_str = base64.b64encode(dbuf).decode('utf-8')

        prompt = (
            "You are an expert veterinary image analyst. I am providing TWO images of the same cropped region of a horse's front leg(s).\n"
            "Image 1: The original RGB image (background removed).\n"
            "Image 2: A Depth Map heatmap of the exact same region (red/orange/yellow pixels are CLOSER to the camera, green/blue pixels are FURTHER away).\n\n"
            
            "The BOTTOM of the images shows the horse's hoof(s). "
            "There may be one front leg, or TWO front legs touching each other. The tail or horse's chest/body may also be visible.\n\n"
            "Your task is to identify the MAIN FRONT LEG. You MUST use the Depth Map to determine which leg is closer to the camera (the red/orange/yellow one). "
            "Additionally, the MAIN leg is almost always the one positioned closest to the HORIZONTAL CENTER of the image (in the middle of the 'x' axis).\n\n"
            
            "Return a JSON object with three keys: 'front_leg_hoof', 'front_leg_fetlock', and 'touching_objects'.\n"
            "- 'front_leg_hoof': a single {y, x} point (normalized 0-1000 scale, y=0 is TOP) inside the thickest part of the MAIN front leg's HOOF (closest to the horizontal center).\n"
            "- 'front_leg_fetlock': a single {y, x} point inside the MAIN front leg's FETLOCK (ankle joint, right above the hoof).\n"
            "- 'touching_objects': a list of {y, x} points for ANY OTHER objects (like a background leg, a tail, or the body/chest) that are touching or overlapping the main front leg.\n\n"
            
            "CRITICAL RULES for touching_objects:\n"
            "1. If there is a SEPARATE object (like a background leg or tail) that is BLUE or GREEN in the Depth Map (further away), you MUST place at least one touching_objects point on it so it can be separated. The background leg is usually positioned further to the side (away from the center).\n"
            "2. WARNING: The main leg is round, so its EDGES may appear green in the Depth Map. Do NOT place a touching_objects point on the green edges of the main leg itself! Only mark distinct, separate objects.\n"
            "3. If the horse's upper chest or body is visible at the very top of the crop, place a touching_objects point there.\n"
            "4. DO NOT place a touching_objects point on the upper part of the main leg just because the color changes (e.g., from a white sock to brown hair). A single leg often has multiple colors.\n"
            "5. If there are no touching objects (it's just one isolated leg), return an empty list.\n\n"
            
            "OUTPUT STRICTLY VALID JSON ONLY. Example format:\n"
            "{\n"
            '  "front_leg_hoof": {"y": 850, "x": 450},\n'
            '  "front_leg_fetlock": {"y": 650, "x": 450},\n'
            '  "touching_objects": [\n'
            '    {"y": 800, "x": 650},\n'
            '    {"y": 300, "x": 500}\n'
            '  ]\n'
            "}"
        )

        # Narrow-leg heuristic bypass to save API calls
        median_w_crop = int(np.median([np.sum(crop_mask[r] > 0) for r in np.unique(np.where(crop_mask > 0)[0])]))
        if median_w_crop < max(40, int(wm * 0.18)):
            logging.info("Depth isolation bypass: Leg blob is extremely narrow (w=%d). Assuming single isolated leg.", median_w_crop)
            return mask

        logging.info("Sending leg crop to Gemini for multi-part point seeds...")
        try:
            import time
            from google.api_core.exceptions import ResourceExhausted
            import os as _os
            
            # Load secondary key from env if available
            _secondary_key = _os.environ.get('GEMINI_API_KEY_2', '')
            
            _models_to_try = [
                'gemini-3.7-flash',
                'gemini-3.6-flash',
                'gemini-3.5-flash-lite',
                'gemini-2.5-flash',
                'gemini-flash-latest'
            ]
            
            # Build list of (api_key, model) pairs to attempt
            _attempts = [(api_key, m) for m in _models_to_try]
            if _secondary_key and _secondary_key != api_key:
                _attempts += [(_secondary_key, m) for m in _models_to_try]
            
            result = None
            for _try_key, _model_name in _attempts:
                try:
                    genai.configure(api_key=_try_key)
                    _key_label = "primary" if _try_key == api_key else "secondary"
                    logging.info("Trying model: %s (%s key)", _model_name, _key_label)
                    _model = genai.GenerativeModel(_model_name, generation_config={"response_mime_type": "application/json"})
                    contents = [{'mime_type': 'image/jpeg', 'data': b64_str}]
                    if depth_b64_str:
                        contents.append({'mime_type': 'image/jpeg', 'data': depth_b64_str})
                    contents.append(prompt)
                    response = _model.generate_content(contents)
                    result = json.loads(response.text)
                    if isinstance(result, list):
                        result = result[0]
                    logging.info("Gemini parsed output (%s/%s): %s", _key_label, _model_name, result)
                    break  # Success
                except ResourceExhausted as _e:
                    # If this is the last attempt overall, we must sleep and retry.
                    # Otherwise, immediately try the next key/model in the loop without waiting.
                    _idx = _attempts.index((_try_key, _model_name))
                    if _idx == len(_attempts) - 1:
                        _retry_s = 60
                        try:
                            import re as _re
                            _m = _re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', str(_e))
                            if _m:
                                _retry_s = int(_m.group(1))
                        except Exception:
                            pass
                        logging.warning("Model %s (%s key) quota exceeded. Last fallback, waiting %ds...", _model_name, _key_label, _retry_s)
                        time.sleep(_retry_s + 2)
                    else:
                        logging.warning("Model %s (%s key) quota exceeded. Proceeding to next model/key immediately...", _model_name, _key_label)
                except (json.JSONDecodeError, Exception) as _je:
                    if '404' in str(_je) or 'not available' in str(_je).lower():
                        logging.warning("Model %s not available, skipping...", _model_name)
                    else:
                        logging.warning("Model %s error (%s), trying next...", _model_name, _je)

            if result is None:
                raise RuntimeError("All Gemini models and keys exhausted.")
            
            # crop_mask and dimensions are already defined at the top of the function

            # Use nobg alpha as pixel boundary if available — it gives finer per-pixel
            # separation than the coarse blob mask (catches semi-transparent 2nd leg too)
            if nobg_bgra is not None:
                crop_alpha = nobg_bgra[c_y_min:c_y_min+ch, c_x_min:c_x_min+cw, 3].copy()
                ws_boundary_mask = (crop_alpha >= 128).astype(np.uint8) * 255
            else:
                ws_boundary_mask = crop_mask

            # Create markers for watershed
            markers = np.zeros_like(ws_boundary_mask, dtype=np.int32)
            
            # Background = pixels fully outside the horse silhouette
            markers[ws_boundary_mask == 0] = 1
            
            # Process the front leg seeds (hoof and fetlock)
            front_leg_label = 2
            
            hoof_obj = result.get('front_leg_hoof', {})
            fetlock_obj = result.get('front_leg_fetlock', {})
            
            hx, hy, fx, fy = -1, -1, -1, -1
            
            if isinstance(hoof_obj, dict) and hoof_obj:
                n_y, n_x = int(hoof_obj.get('y', -1)), int(hoof_obj.get('x', -1))
                if n_y != -1 and n_x != -1:
                    hy = int((n_y / 1000.0) * ch)
                    hx = int((n_x / 1000.0) * cw)
                    cv2.circle(markers, (hx, hy), 15, front_leg_label, -1)
            
            if isinstance(fetlock_obj, dict) and fetlock_obj:
                n_y, n_x = int(fetlock_obj.get('y', -1)), int(fetlock_obj.get('x', -1))
                if n_y != -1 and n_x != -1:
                    fy = int((n_y / 1000.0) * ch)
                    fx = int((n_x / 1000.0) * cw)
                    cv2.circle(markers, (fx, fy), 15, front_leg_label, -1)
            
            # If both were provided, draw a thick line between them to seed the core up to the fetlock
            if hy != -1 and fy != -1:
                cv2.line(markers, (hx, hy), (fx, fy), front_leg_label, 10)
            
            # Process all touching background objects — each gets its own label
            background_label = 3
            touching_objects = result.get('touching_objects', [])
            if isinstance(touching_objects, list):
                for point_obj in touching_objects:
                    if not isinstance(point_obj, dict):
                        continue
                    n_y, n_x = int(point_obj.get('y', -1)), int(point_obj.get('x', -1))
                    if n_y != -1 and n_x != -1:
                        cy = int((n_y / 1000.0) * ch)
                        cx = int((n_x / 1000.0) * cw)
                        cv2.circle(markers, (cx, cy), 15, background_label, -1)
                        background_label += 1

            # Use the ORIGINAL photo (not the white-bg nobg crop) for watershed gradients
            # so the algorithm can trace real photographic edges between the two legs.
            crop_orig = original_bgr[c_y_min:c_y_min+ch, c_x_min:c_x_min+cw].copy()
            crop_bgr_ws = cv2.GaussianBlur(crop_orig, (5, 5), 0)
            
            # Blend the depth heatmap into the watershed image to force it to snap to depth boundaries!
            if depth_color is not None and depth_color.shape == crop_bgr_ws.shape:
                crop_bgr_ws = cv2.addWeighted(crop_bgr_ws, 0.4, depth_color, 0.6, 0)
            
            # Mask out true background pixels so watershed boundary is respected
            crop_bgr_ws[ws_boundary_mask == 0] = 0
            
            cv2.watershed(crop_bgr_ws, markers)
            
            # Keep only the front_leg region
            final_crop_mask = np.zeros_like(ws_boundary_mask)
            final_crop_mask[markers == front_leg_label] = 255
            
            # Smooth the isolated mask to prevent broken lines during axis calculation
            kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            final_crop_mask = cv2.morphologyEx(final_crop_mask, cv2.MORPH_CLOSE, kernel_smooth)
            final_crop_mask = cv2.morphologyEx(final_crop_mask, cv2.MORPH_OPEN, kernel_smooth)
            
            # --- TRIM ABOVE FETLOCK ---
            # The user requested to cut the leg mask above the fetlock to include a little cannon bone.
            if fy != -1 and hy != -1:
                # Calculate the vertical distance between the hoof and the fetlock
                dy = max(10, hy - fy)
                # Go up from the fetlock by that same distance to reach the lower cannon bone
                trim_y = max(0, fy - dy)
                final_crop_mask[:trim_y, :] = 0
                logging.info(f"Trimmed mask in lower cannon bone (crop y={trim_y}, dy={dy})")
            
            refined_mask = np.zeros_like(mask)
            refined_mask[c_y_min:c_y_min+ch, c_x_min:c_x_min+cw] = final_crop_mask
            
            logging.info("Watershed successfully isolated the front leg using Gemini seeds.")
            
            return refined_mask
        except Exception as e:
            logging.error("Failed to refine mask with Gemini: %s", e)
            return mask

    for i, info in enumerate(leg_infos):
        mask_to_use = info['mask']
        
        if gemini_key:
            # Pass the FULL original mask and depth map to Gemini so the crop includes the entire leg
            mask_to_use = refine_mask_with_gemini(img, mask, gemini_key, nobg_bgra=processed_bgra, depth_map=depth_map)
            if do_debug:
                cv2.imwrite(str(p.parent / f"{p.stem}_gemini_refined_leg_{i}.png"), mask_to_use)

        pt_top, pt_bottom = find_cannon_bone_axis(
            mask_to_use,
            target_knee=info.get('knee'),
            target_hoof=info.get('hoof'),
        )
        green, red, dominant = analyze_symmetry(mask_to_use, pt_top, pt_bottom)
        combined_green = np.maximum(combined_green, green)
        combined_red = np.maximum(combined_red, red)
        per_leg_draw.append({'pt_top': pt_top, 'pt_bottom': pt_bottom, 'dominant': dominant})

    # --- Final output: overlay is applied on the ORIGINAL image ---
    out = apply_overlay(img, combined_green, combined_red, alpha=0.55)

    for i, di in enumerate(per_leg_draw):
        cv2.line(out, di['pt_top'], di['pt_bottom'], (255, 80, 0), max(2, int(w * 0.004)))
        cv2.putText(out, f"Leg {i + 1} Dominant: {di['dominant']}",
                    (10, 30 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite(str(p.parent / f"{p.stem}_analyzed.jpg"), out)

    if do_debug:
        dbg = img.copy()
        dbg[combined_green > 0] = [34, 197, 94]
        dbg[combined_red > 0] = [48, 48, 220]
        for di in per_leg_draw:
            cv2.line(dbg, di['pt_top'], di['pt_bottom'], (255, 80, 0), max(2, int(w * 0.004)))
        cv2.imwrite(str(p.parent / f"{p.stem}_debug.png"), dbg)

    logging.info("Saved %s_analyzed.jpg", p.stem)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Horse leg symmetry analyzer (v2) — requires original and pre-removed background images."
    )
    parser.add_argument(
        "original_image",
        help="Path to the original image.",
    )
    parser.add_argument(
        "processed_image",
        help="Path to the processed (background removed) image.",
    )
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    parser.add_argument("--use-ai", action="store_true",
                        help="Enable MMPose AI keypoint detection if available")
    parser.add_argument("--gemini-key", type=str, default=None,
                        help="Optional Gemini API key for VLM mask refinement")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Optional local model path for MMPoseInferencer")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for MMPose (e.g. cpu or cuda:0)")
    args = parser.parse_args()

    inferencer = None
    if args.use_ai and MMPoseInferencer is not None:
        try:
            kwargs = {"device": args.device} if args.device else {}
            pose = args.model_path or 'rtmpose-m_8xb64-210e_ap10k-256x256'
            inferencer = MMPoseInferencer(pose2d=pose, **kwargs)
            logging.info("MMPose inferencer initialized.")
        except Exception as e:
            logging.warning("Failed to initialize MMPoseInferencer: %s", e)

    try:
        process_image(args.original_image, args.processed_image, do_debug=args.debug, inferencer=inferencer, gemini_key=args.gemini_key)
    except Exception as e:
        logging.exception("Failed processing %s: %s", args.original_image, e)


if __name__ == "__main__":
    main()