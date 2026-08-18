
from pathlib import Path
import glob
import argparse
import logging
import cv2
import numpy as np
import sys
from PIL import Image as PILImage
from transformers import pipeline as hf_pipeline
from mmpose.apis import MMPoseInferencer
import threading
import json
import base64

try:
    import google.generativeai as genai
except ImportError:
    genai = None

_depth_lock = threading.Lock()


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_depth_pipe = None


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

def get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        import torch
        device_id = 0 if torch.cuda.is_available() else -1
        logging.info(f"Loading Depth Anything V2 Small on device {device_id} …")
        _depth_pipe = hf_pipeline("depth-estimation", model=_DEPTH_MODEL_ID, device=device_id)
        logging.info("Depth Anything V2 ready.")
    return _depth_pipe


def estimate_depth(image_bgr: np.ndarray,
                   fg_mask: np.ndarray | None = None) -> np.ndarray:
    """Depth Anything V2 → float32 depth map [0, 1].  1.0 = closest.

    FIX #1: Background pixels are filled with neutral gray (127) before
    inference so the model is not confused by black zeros.  After inference,
    depth outside fg_mask is zeroed so only horse pixels contribute to scoring.
    """
    h, w = image_bgr.shape[:2]

    if fg_mask is not None:
        input_img = image_bgr.copy()
        input_img[fg_mask == 0] = 127
    else:
        input_img = image_bgr

    logging.info("Running Depth Anything V2 …")
    rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    
    with _depth_lock:
        result = get_depth_pipe()(pil_img)
        
    depth_np = np.array(result["depth"]).astype(np.float32)

    if depth_np.shape[0] != h or depth_np.shape[1] != w:
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

    d_min, d_max = depth_np.min(), depth_np.max()
    depth_np = (depth_np - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_np)

    if fg_mask is not None:
        depth_np[fg_mask == 0] = 0.0

    logging.info("Depth map ready (shape=%s).", depth_np.shape)
    return depth_np


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


# ---------------------------------------------------------------------------
# FIX #9 (ACTIVATED) — depth-based foreground prefilter
# ---------------------------------------------------------------------------

def depth_prefilter_mask(mask: np.ndarray,
                          depth_map: np.ndarray,
                          depth_delta: float = 0.27) -> np.ndarray:
    """Remove far/back-leg pixels from the foreground mask before leg selection.

    Keeps foreground pixels whose depth >= the Nth percentile of foreground
    depths.  In front-on shots this drops the back leg (blue in TURBO) while
    keeping the front leg (orange/red).

    FIX #12 — percentile lowered from 50 → 25.
    When the camera is at ground level pointing upward the hoof is the closest
    point (depth ≈ 1.0) while the cannon bone is further away (depth ≈ 0.4–0.6).
    A 50th-percentile threshold sits exactly at the hoof/cannon-bone boundary,
    stripping the cannon bone entirely.  25th-percentile keeps the vast majority
    of the front leg while still discarding the clearly-far back leg pixels.

    FIX #13 — height-preservation safety guard.
    The existing pixel-count guard (< 15 %) does not catch the case where the
    top of the leg is cut off (cannon bone has low pixel count relative to the
    wide hoof).  An additional check compares the bounding-box HEIGHT of the
    filtered mask to the original: if height shrinks by more than 35 % the
    filter is discarding the top of the leg, so the original is returned.
    """
    fg_depths = depth_map[mask > 0]
    if fg_depths.size == 0:
        return mask

    # Record original bounding-box height for the height-safety check
    ys_orig = np.where(np.any(mask > 0, axis=1))[0]
    orig_height = int(ys_orig[-1] - ys_orig[0]) if ys_orig.size >= 2 else 0

    farthest_depth = float(fg_depths.min())
    thresh = farthest_depth + depth_delta
    filtered = np.zeros_like(mask)
    filtered[(mask > 0) & (depth_map >= thresh)] = 255

    # Close small gaps so the kept region stays contiguous
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, k)

    # Safety guard 1: pixel count (original)
    if cv2.countNonZero(filtered) < cv2.countNonZero(mask) * 0.15:
        logging.warning("depth_prefilter_mask: pixel count too small — returning original mask.")
        return mask

    # Safety guard 2 (FIX #13): height preservation
    # If the filter is cutting off the top of the leg the bounding-box height
    # shrinks.  More than 35 % shrinkage means the cannon bone is being lost.
    if orig_height > 0:
        ys_filt = np.where(np.any(filtered > 0, axis=1))[0]
        if ys_filt.size >= 2:
            filt_height = int(ys_filt[-1] - ys_filt[0])
            if filt_height < orig_height * 0.65:
                logging.warning(
                    "depth_prefilter_mask: height shrank to %.0f%% (orig=%d filt=%d) "
                    "— top of leg cut off; returning original mask.",
                    100.0 * filt_height / orig_height, orig_height, filt_height)
                return mask

    logging.info("depth_prefilter_mask: kept %.1f%% of fg pixels (thresh depth=%.3f)",
                 100.0 * cv2.countNonZero(filtered) / max(1, cv2.countNonZero(mask)), thresh)
    return filtered


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
                               depth_map: np.ndarray | None = None,
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

        avg_depth = 0.0
        if depth_map is not None:
            fg_px = part > 0
            if np.any(fg_px):
                avg_depth = float(depth_map[fg_px].mean())

        if debug:
            logging.info("Candidate: area=%.0f bottom_w=%d dx=%.1f depth=%.3f",
                         area, bottom_width, dx, avg_depth)
        candidates.append({
            'mask': part, 'area': area,
            'bottom_width': bottom_width, 'dx': dx, 'avg_depth': avg_depth,
        })

    if not candidates:
        return None

    max_area = max(c['area'] for c in candidates)
    valid_candidates = [c for c in candidates if c['area'] >= max_area * 0.15]
    valid_candidates.sort(key=lambda c: (c['avg_depth'], c['area']), reverse=True)
    best = valid_candidates[0]
    logging.info("Selected front leg candidate blob (area=%.0f depth=%.3f) for Gemini", best['area'], best['avg_depth'])
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

    # Build an BGR view of the processed image for depth estimation
    # Background pixels are already transparent; we fill them with neutral
    # gray (127) so Depth Anything V2 is not biased by black zeros.
    fg_bgr = processed_bgra[:, :, :3].copy()
    fg_bgr[mask == 0] = 127

    # --- Depth estimation (FIX #1: neutral-gray background fill) ---
    depth_map = estimate_depth(fg_bgr, fg_mask=mask)
    depth_color = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(p.parent / f"{p.stem}_depth.png"), depth_color)
    logging.info("Saved depth map.")

    # --- Depth prefilter: remove far/back-leg pixels (FIX #9 / #12 / #13) ---
    depth_mask = depth_prefilter_mask(mask, depth_map, depth_delta=0.27)
    if do_debug:
        cv2.imwrite(str(p.parent / f"{p.stem}_depth_mask.png"), depth_mask)
        logging.info("Saved depth-filtered mask (debug).")

    leg_masks = []
    leg_infos = []

    # --- AI path (MMPose keypoints on the original image) ---
    if inferencer is not None:
        try:
            legs = get_ai_leg_keypoints(inferencer, str(p))
        except Exception:
            legs = []
        if legs:
            best_leg, best_depth_val = None, -1.0
            for knee, hoof in legs:
                line_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(line_mask,
                         (int(knee[0]), int(knee[1])),
                         (int(hoof[0]), int(hoof[1])), 255, thickness=5)
                overlap = (line_mask > 0) & (mask > 0)
                avg_d = float(depth_map[overlap].mean()) if np.any(overlap) else 0.0
                logging.info("Leg candidate knee=(%.1f,%.1f) hoof=(%.1f,%.1f) depth=%.3f",
                             *knee, *hoof, avg_d)
                if avg_d > best_depth_val:
                    best_depth_val, best_leg = avg_d, (knee, hoof)

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
        lm = select_front_leg_fallback(depth_mask, depth_map=depth_map, debug=do_debug)
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

    def refine_mask_with_gemini(original_bgr: np.ndarray, mask: np.ndarray, api_key: str, nobg_bgra: np.ndarray = None) -> np.ndarray:
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

        _, buffer = cv2.imencode('.jpg', crop_bgr)
        b64_str = base64.b64encode(buffer).decode('utf-8')

        prompt = (
            "You are an expert veterinary image analyst. This image shows a cropped region of a horse. "
            "The TOP of the image shows the horse's tail or body hair. "
            "The BOTTOM of the image shows the horse's front leg(s) and hoof(s). "
            "Focus ONLY on the BOTTOM half of the image where the legs and hooves are visible. "
            "There may be one front leg, or TWO front legs touching each other, and possibly a tail also touching. "
            "Return a JSON object with two keys: 'front_leg' and 'touching_objects'. "
            "'front_leg' must be a single {y, x} point (in normalized 0 to 1000 scale, where y=0 is TOP and y=1000 is BOTTOM) "
            "placed safely inside the thickest part of the MAIN front leg (closest to horizontal center), in the lower half of the image (y > 500). "
            "'touching_objects' must be a LIST of {y, x} points. For EACH distinct object touching the main front leg "
            "(a second front leg, tail hair, back leg), place one point safely inside that object, also in the lower portion (y > 400). "
            "CRITICAL: If you see TWO separate leg/hoof shapes at the bottom, they MUST each get their own seed point. "
            "If nothing is touching the main front leg, 'touching_objects' should be an empty list.\n\n"
            "OUTPUT STRICTLY VALID JSON ONLY. Example format:\n"
            "{\n"
            '  "front_leg": {"y": 700, "x": 450},\n'
            '  "touching_objects": [\n'
            '    {"y": 650, "x": 650},\n'
            '    {"y": 300, "x": 500}\n'
            '  ]\n'
            "}"
        )

        logging.info("Sending leg crop to Gemini for multi-part point seeds...")
        try:
            import time
            from google.api_core.exceptions import ResourceExhausted
            import os as _os
            
            # Load secondary key from env if available
            _secondary_key = _os.environ.get('GEMINI_API_KEY_2', '')
            
            # Waterfall: try all models on primary key, then all models on secondary key
            _models_to_try = [
                'gemini-3.6-flash',
                'gemini-3.7-flash',
                'gemini-3.5-flash',
                'gemini-3.1-pro-preview',
                'gemini-3.5-flash-lite',
                'gemini-3.1-flash-lite',
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
                    response = _model.generate_content([
                        {'mime_type': 'image/jpeg', 'data': b64_str},
                        prompt
                    ])
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
            
            ch, cw = crop_bgr.shape[:2]
            crop_mask = mask[c_y_min:c_y_min+ch, c_x_min:c_x_min+cw].copy()

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
            
            # Process the front leg seed
            front_leg_label = 2
            front_leg_obj = result.get('front_leg', {})
            if isinstance(front_leg_obj, dict) and front_leg_obj:
                n_y, n_x = int(front_leg_obj.get('y', -1)), int(front_leg_obj.get('x', -1))
                if n_y != -1 and n_x != -1:
                    cy = int((n_y / 1000.0) * ch)
                    cx = int((n_x / 1000.0) * cw)
                    cv2.circle(markers, (cx, cy), 15, front_leg_label, -1)
            
            # Anchor seed: place a background label in the top strip of the crop (horse body above the leg).
            # This stops watershed from flooding upward into the torso when touching_objects seeds
            # are all placed near the hoof area (same y-level as the front_leg seed).
            body_label = 100  # Use a high label to not conflict with touching_objects
            top_strip_y = max(0, int(ch * 0.05))  # 5% from the top
            mid_x = cw // 2
            # Find a pixel inside the silhouette in the top strip
            if ws_boundary_mask[top_strip_y, mid_x] > 0:
                cv2.circle(markers, (mid_x, top_strip_y), 10, body_label, -1)
            else:
                # Try a few columns to find one inside the silhouette
                for test_x in [cw // 4, cw // 3, cw * 2 // 3, cw * 3 // 4]:
                    if ws_boundary_mask[top_strip_y, test_x] > 0:
                        cv2.circle(markers, (test_x, top_strip_y), 10, body_label, -1)
                        break
            
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
            
            # Mask out true background pixels so watershed boundary is respected
            crop_bgr_ws[ws_boundary_mask == 0] = 0
            
            cv2.watershed(crop_bgr_ws, markers)
            
            # Keep only the front_leg region
            final_crop_mask = np.zeros_like(ws_boundary_mask)
            final_crop_mask[markers == front_leg_label] = 255
            
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
            mask_to_use = refine_mask_with_gemini(img, mask_to_use, gemini_key, nobg_bgra=processed_bgra)
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