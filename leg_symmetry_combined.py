
from pathlib import Path
import glob
import argparse
import logging
import cv2
import numpy as np
import sys
from PIL import Image as PILImage
from transformers import pipeline as hf_pipeline
try:
    from mmpose.apis import MMPoseInferencer
except ImportError:
    MMPoseInferencer = None
import threading

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
# FIX #7 — split_mask_on_width (module level)
# ---------------------------------------------------------------------------

def split_mask_on_width(mask_region: np.ndarray,
                        min_narrow_frac: float = 0.6,
                        min_width_px: int = 30,
                        debug: bool = False) -> list[np.ndarray]:
    """Split a mask at narrow 'waist' rows; return component masks sorted by area desc."""
    ys = np.where(np.any(mask_region > 0, axis=1))[0]
    if ys.size == 0:
        return [mask_region]
    y0, y1 = int(ys[0]), int(ys[-1])

    widths = np.zeros(y1 - y0 + 1, dtype=np.int32)
    for i, ry in enumerate(range(y0, y1 + 1)):
        xs = np.where(mask_region[ry] > 0)[0]
        widths[i] = int(xs[-1] - xs[0]) if xs.size >= 2 else 0

    nonzero = widths[widths > 0]
    if nonzero.size == 0:
        return [mask_region]
    median_w = int(np.median(nonzero))
    thresh = max(min_width_px, int(median_w * min_narrow_frac))

    narrow = widths < thresh
    cut_rows = []
    i = 0
    while i < len(narrow):
        if narrow[i]:
            j = i
            while j + 1 < len(narrow) and narrow[j + 1]:
                j += 1
            cut_rows.append(y0 + (i + j) // 2)
            i = j + 1
        else:
            i += 1

    if not cut_rows:
        return [mask_region]

    split = mask_region.copy()
    pad = 2
    for r in cut_rows:
        split[max(y0, r - pad): min(y1, r + pad) + 1, :] = 0

    num_labels, labels = cv2.connectedComponents(split)
    parts = []
    for lab in range(1, num_labels):
        part = np.zeros_like(mask_region)
        part[labels == lab] = 255
        if cv2.countNonZero(part) > 50:
            parts.append(part)

    if not parts:
        return [mask_region]
    parts.sort(key=lambda m: cv2.countNonZero(m), reverse=True)
    if debug:
        logging.info("split_mask_on_width: %d parts (median_w=%d thresh=%d cuts=%s)",
                     len(parts), median_w, thresh, cut_rows)
    return parts


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
# FIX #3 — tail rejection helper
# ---------------------------------------------------------------------------

def is_likely_tail(contour: np.ndarray, mask: np.ndarray) -> bool:
    """Return True if contour resembles a tail rather than a leg."""
    x, y, cw, ch = cv2.boundingRect(contour)
    h_img, w_img = mask.shape
    if ch == 0:
        return False

    aspect = ch / max(cw, 1)
    if aspect > 6 and cw < w_img * 0.07:
        return True

    top_end = y + max(1, int(ch * 0.20))
    bot_start = y + int(ch * 0.80)
    top_ws, bot_ws = [], []
    for ry in range(y, min(top_end + 1, h_img)):
        xs = np.where(mask[ry] > 0)[0]
        if xs.size >= 2:
            top_ws.append(int(xs[-1] - xs[0]))
    for ry in range(bot_start, min(y + ch + 1, h_img)):
        xs = np.where(mask[ry] > 0)[0]
        if xs.size >= 2:
            bot_ws.append(int(xs[-1] - xs[0]))

    if top_ws and bot_ws:
        avg_top = float(np.mean(top_ws))
        avg_bot = float(np.mean(bot_ws))
        if avg_bot < avg_top * 0.45 and avg_bot < w_img * 0.04:
            return True

    return False


# ---------------------------------------------------------------------------
# Depth-based continuous-leg isolation (avoids fetlock-waist fragmentation)
# ---------------------------------------------------------------------------

def isolate_legs_from_depth(depth_mask: np.ndarray,
                            depth_map: np.ndarray,
                            min_area_frac: float = 0.01,
                            min_height_frac: float = 0.25,
                            debug: bool = False) -> list[np.ndarray]:
    """Isolate one or more continuous front-leg blobs straight from the
    depth-filtered foreground mask, WITHOUT splitting at narrow waists.

    select_front_leg_fallback() calls split_mask_on_width(), which cuts the
    mask wherever it narrows — but a real leg narrows naturally at the
    fetlock/pastern. On ground-level shots (hoof closer to the camera than
    the cannon bone) that cut produces two pieces, and the candidate scorer
    picks the hoof piece (higher avg depth) over the cannon piece, losing
    the upper leg entirely. This function instead keeps each leg as a single
    connected component, so the fetlock narrowing stays part of one shape.

    FIX: Apply a vertical ROI — exclude the top 30% of the image (horse body,
    blanket, torso) before looking for leg contours.  Legs always start
    below that band; the blanket/body regions at similar depth that were
    bleeding into the mask live in the top portion.  We also require candidate
    blobs to be taller than they are wide (aspect ratio > 0.8) so wide body
    blobs are rejected even if they somehow extend below the cutoff.
    """
    h, w = depth_mask.shape

    # --- Vertical ROI: only look in the lower 70% of the frame ---
    roi_top = int(h * 0.30)
    roi_mask = depth_mask.copy()
    roi_mask[:roi_top, :] = 0

    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Fall back to full mask if ROI yields nothing
        cnts, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = depth_mask

    if not cnts:
        return []

    min_area = h * w * min_area_frac
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh < h * min_height_frac:
            if debug:
                logging.info("isolate_legs_from_depth: rejected short blob (h=%d, area=%.0f)", bh, area)
            continue
        # Reject blobs that are wider than tall (body/blanket regions)
        if bw > 0 and (bh / bw) < 0.8:
            if debug:
                logging.info("isolate_legs_from_depth: rejected wide blob (h=%d w=%d ratio=%.2f)", bh, bw, bh/bw)
            continue
        part = np.zeros_like(depth_mask)
        cv2.drawContours(part, [cnt], -1, 255, cv2.FILLED)
        if is_likely_tail(cnt, part):
            if debug:
                logging.info("isolate_legs_from_depth: rejected tail-like contour (area=%.0f)", area)
            continue
        candidates.append((part, area))

    candidates.sort(key=lambda t: t[1], reverse=True)
    result = [m for m, _ in candidates]
    if debug:
        logging.info("isolate_legs_from_depth: found %d candidate leg(s).", len(result))
    return result


# ---------------------------------------------------------------------------
# Leg selection — fallback (no MMPose)
# ---------------------------------------------------------------------------

def select_front_leg_fallback(mask: np.ndarray,
                               depth_map: np.ndarray | None = None,
                               debug: bool = False) -> np.ndarray | None:
    """Select the frontmost front leg using depth + heuristics.

    Receives depth-filtered mask (FIX #9) so back-leg pixels are already
    removed before this function runs.
    """
    h, w = mask.shape
    zone = np.zeros_like(mask)
    # FIX #14: zone cutoff lowered from 35% → 20% of image height.
    # At 35% the cannon bone (which starts at ~20-30% from the top in
    # ground-level shots) was sometimes excluded from the candidate region.
    zone[int(h * 0.20):] = mask[int(h * 0.20):]

    raw_contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not raw_contours:
        return None

    combined = np.zeros_like(mask)
    for cnt in raw_contours:
        if cv2.contourArea(cnt) >= 500:
            cv2.drawContours(combined, [cnt], -1, 255, cv2.FILLED)

    parts = split_mask_on_width(combined,
                                min_narrow_frac=0.55,
                                min_width_px=max(20, int(0.05 * w)),
                                debug=debug)

    if len(parts) == 1 and cv2.countNonZero(parts[0]) > (h * w * 0.03):
        hooves = []
        for cnt in raw_contours:
            if cv2.contourArea(cnt) < 500:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            hooves.append((float(bx + bw // 2), float(min(by + bh + 20, h - 1))))
        if len(hooves) >= 2:
            ws_parts = seed_watershed_from_hooves(combined, hooves)
            ws_parts = [p for p in ws_parts if p is not None and cv2.countNonZero(p) > 50]
            if len(ws_parts) > len(parts):
                logging.info("Watershed separated %d parts from touching-leg blob", len(ws_parts))
                parts = ws_parts

    img_cx = w / 2.0
    candidates = []
    for part in parts:
        cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        if is_likely_tail(cnt, part):
            if debug:
                logging.info("Rejected tail-like contour (area=%.0f)", area)
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx_part = bx + bw / 2.0
        dx = abs(cx_part - img_cx)

        bottom_width = 0
        for ry in range(by + int(bh * 0.75), min(by + bh + 1, h)):
            xs = np.where(part[ry] > 0)[0]
            if xs.size > 0:
                bottom_width = max(bottom_width, int(xs[-1] - xs[0]))

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

    candidates = [c for c in candidates
                  if c['area'] >= 800 and c['bottom_width'] >= max(20, int(0.12 * w))]
    if not candidates:
        return None

    candidates.sort(key=lambda c: (c['avg_depth'], c['area']), reverse=True)
    best = candidates[0]
    logging.info("Selected front leg (area=%.0f depth=%.3f)", best['area'], best['avg_depth'])
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

    if cv2.countNonZero(lm) > (mask.shape[0] * mask.shape[1] * 0.02):
        parts = split_mask_on_width(lm,
                                    min_narrow_frac=0.6,
                                    min_width_px=max(20, int(0.06 * mask.shape[1])),
                                    debug=debug)
        if len(parts) > 1:
            kx = int(round(knee[0]))
            best_part = min(
                parts,
                key=lambda p: abs(int(np.mean(np.where(p > 0)[1])) - kx)
                              if np.any(p > 0) else float('inf')
            )
            lm = best_part

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
# Watershed leg separator
# ---------------------------------------------------------------------------

def seed_watershed_from_hooves(mask: np.ndarray,
                                hooves: list[tuple[float, float]],
                                rgba: np.ndarray | None = None,
                                snap_radius: int = 30) -> list:
    """Segment mask into regions seeded at each hoof point using watershed."""
    if mask is None or mask.size == 0:
        return []
    bm = (mask > 0).astype(np.uint8) * 255
    h, w = bm.shape

    def snap_to_mask(xf, yf):
        x, y = int(round(xf)), int(round(yf))
        if 0 <= x < w and 0 <= y < h and bm[y, x] > 0:
            return (x, y)
        for r in range(1, snap_radius + 1):
            best, bestd = None, None
            for yy in range(max(0, y - r), min(h, y + r + 1)):
                for xx in range(max(0, x - r), min(w, x + r + 1)):
                    if bm[yy, xx] > 0:
                        d = (xx - x) ** 2 + (yy - y) ** 2
                        if bestd is None or d < bestd:
                            bestd, best = d, (xx, yy)
            if best:
                return best
        return None

    markers = np.zeros((h, w), dtype=np.int32)
    seed_points = []
    for i, (kx, ky) in enumerate(hooves, start=1):
        s = snap_to_mask(kx, ky)
        seed_points.append(s)
        if s:
            cv2.circle(markers, s, 6, i, -1)

    if all(s is None for s in seed_points):
        return []

    try:
        if rgba is not None:
            gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)
            inv = (255 - gray).astype(np.float32) / 255.0
            dist = cv2.distanceTransform((bm // 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            if dist.max() <= 0:
                return []
            topo = dist * (1.0 + 0.7 * inv)
            topo8 = np.uint8((topo / topo.max()) * 255.0)
            img3 = cv2.cvtColor(topo8, cv2.COLOR_GRAY2BGR)
        else:
            dist = cv2.distanceTransform((bm // 255).astype(np.uint8), cv2.DIST_L2, 5)
            if dist.max() <= 0:
                return []
            dist8 = np.uint8((dist / dist.max()) * 255.0)
            img3 = cv2.cvtColor(dist8, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img3, markers)
    except Exception as e:
        logging.warning("watershed failed: %s", e)
        return []

    parts = []
    for i in range(1, len(hooves) + 1):
        m = np.zeros_like(bm)
        m[markers == i] = 255
        parts.append(m if cv2.countNonZero(m) > 50 else None)
    return parts


# ---------------------------------------------------------------------------
# MMPose helpers
# ---------------------------------------------------------------------------

def get_ai_leg_keypoints(inferencer,
                         image_path: str,
                         bboxes: list[list[float]] | None = None) -> list[dict]:
    if inferencer is None:
        return []
    try:
        if bboxes:
            res = inferencer(image_path, bboxes=bboxes)
        else:
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
            
        def sc(i):
            return float(scores[i]) if scores and len(scores) > i else 1.0

        if len(kpts) == 4:
            pts = [tuple(k[:2]) for k in kpts]
            scores_list = [sc(i) for i in range(4)]
            logging.info(f"AI Model Raw Points: {pts}")
            logging.info(f"AI Model Raw Scores: {scores_list}")
            legs.append({
                'type': '4kp',
                'knee': pts[0],
                'hoof': pts[3],
                'ai_keypoints': pts,
                'ai_scores': scores_list
            })
        elif len(kpts) > 10:
            if sc(6) > 0.12 and sc(7) > 0.12:
                legs.append({
                    'type': 'ap10k',
                    'knee': tuple(kpts[6][:2]),
                    'hoof': tuple(kpts[7][:2])
                })
            if sc(9) > 0.12 and sc(10) > 0.12:
                legs.append({
                    'type': 'ap10k',
                    'knee': tuple(kpts[9][:2]),
                    'hoof': tuple(kpts[10][:2])
                })
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
    line is always at 90d from the ground — no diagonal slant.

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


def find_hoof_top_robust(leg_mask: np.ndarray) -> tuple[int, int, int]:
    """Finds hoof top Y, base Y, and bottom Y from the leg mask.
    
    Returns:
        (hoof_top_y, base_y, bottom_y)
    """
    h, w = leg_mask.shape
    ys = np.where(np.any(leg_mask > 0, axis=1))[0]
    if ys.size == 0:
        return 0, 0, 0
    top_y = int(ys[0])
    bottom_y = int(ys[-1])
    
    # Extract left and right boundaries
    L = {}
    R = {}
    for y in range(top_y, bottom_y + 1):
        xs = np.where(leg_mask[y] > 0)[0]
        if xs.size >= 2:
            L[y] = xs[0]
            R[y] = xs[-1]
            
    # Scan from bottom-up (ignoring the bottom 15 pixels of ground noise) to find any sudden jump
    jump_y = None
    for y in range(bottom_y - 15, top_y, -1):
        if y in L and (y + 3) in L:
            diff_L = abs(L[y] - L[y + 3])
            diff_R = abs(R[y] - R[y + 3])
            # A jump of > 20 pixels in 3 vertical pixels is a real outlier
            if diff_L > 20 or diff_R > 20:
                jump_y = y
                break
                
    # Define search range for hoof base and hoof top
    search_bottom_y = bottom_y
    if jump_y is not None:
        search_top_y = jump_y + 5
    else:
        # No jump: search within the bottom 30% of the mask
        search_top_y = bottom_y - int((bottom_y - top_y) * 0.30)
        
    # We want to measure widths and find the local minimum in the valid search range
    y_coords = list(range(search_top_y, search_bottom_y + 1))
    widths = []
    for y in y_coords:
        if y in L and y in R:
            widths.append(R[y] - L[y])
        else:
            widths.append(0)
            
    widths = np.array(widths, dtype=np.float64)
    y_coords = np.array(y_coords)
    
    # Smooth the widths
    window = 11
    if len(widths) > window:
        smoothed = np.convolve(widths, np.ones(window)/window, mode='same')
    else:
        smoothed = widths.copy()
        
    # Find widest point (base of hoof) near the bottom
    bottom_15_pct = int(len(widths) * 0.15)
    if bottom_15_pct == 0:
        bottom_15_pct = 1
    base_idx = len(widths) - bottom_15_pct + np.argmax(smoothed[-bottom_15_pct:])
    base_y = y_coords[base_idx]
    
    # Scan upwards from base_idx to find the first local minimum
    hoof_top_y = None
    for i in range(base_idx, 0, -1):
        val = smoothed[i]
        is_min = True
        radius = min(8, i, len(smoothed) - i - 1)
        if radius < 3:
            continue
        for offset in range(1, radius + 1):
            if smoothed[i - offset] < val or smoothed[i + offset] < val:
                is_min = False
                break
        if is_min:
            hoof_top_y = y_coords[i]
            break
            
    # Fallback if no local minimum is found
    if hoof_top_y is None:
        base_width = widths[base_idx]
        fallback_height = int(base_width * 0.45)
        hoof_top_y = base_y - fallback_height
        if hoof_top_y < search_top_y:
            hoof_top_y = search_top_y
            
    return hoof_top_y, base_y, bottom_y


def find_leg_landmarks(leg_mask: np.ndarray) -> dict | None:
    """Locate P1 (top of cannon), P2 (fetlock joint), P3 (coronet band), and
    P4 (hoof base) from a single continuous leg silhouette via a width
    profile scan.

    The leg's width is roughly constant down the cannon bone, narrows to a
    real local minimum at the fetlock/pastern, then flares back out sharply
    as the hoof capsule begins (coronet band) and widens further to its
    base at the ground. This locates BOTH narrow points independently,
    instead of approximating P2 as a plain midpoint.

    FIX: Use a smoothed centerline for X coordinates instead of raw per-row
    center_x(), so the fetlock/coronet markers don't jump laterally due to
    mask noise at narrow rows.  The smoothed centerline also allows us to
    measure lateral deviation of P2/P3/P4 from the P1→P4 plumb line.

    Returns a dict with 'p1_top', 'p2_fetlock', 'p3_coronet', 'p4_hoof'
    (x, y) tuples and deviation metrics, or None if mask is degenerate.
    """
    h, w = leg_mask.shape
    ys = np.where(np.any(leg_mask > 0, axis=1))[0]
    if ys.size < 10:
        return None
    top_y, bottom_y = int(ys[0]), int(ys[-1])

    y_coords = np.arange(top_y, bottom_y + 1)
    n = len(y_coords)

    # Raw per-row width and centre X
    raw_cx = np.full(n, w / 2.0, dtype=np.float64)
    widths  = np.zeros(n, dtype=np.float64)
    for i, y in enumerate(y_coords):
        xs = np.where(leg_mask[y] > 0)[0]
        if xs.size >= 2:
            widths[i]  = float(xs[-1] - xs[0])
            raw_cx[i]  = (float(xs[0]) + float(xs[-1])) / 2.0

    # Smooth both width AND centerline to reduce noise
    window = max(11, n // 20)
    if n > window:
        smoothed = np.convolve(widths,  np.ones(window) / window, mode='same')
        smooth_cx = np.convolve(raw_cx, np.ones(window) / window, mode='same')
    else:
        smoothed   = widths.copy()
        smooth_cx  = raw_cx.copy()

    def scx(idx: int) -> int:
        """Smoothed centre X at index idx."""
        return int(round(float(smooth_cx[np.clip(idx, 0, n - 1)])))

    # --- P4: hoof base — widest point within the bottom 15% of the leg ---
    bottom_band = max(1, int(n * 0.15))
    base_idx = (n - bottom_band) + int(np.argmax(smoothed[-bottom_band:]))

    # --- Pastern: narrowest point above hoof base ---
    pastern_idx = None
    search_radius = max(15, int(n * 0.08))  # ~8% of leg height ignores small hair bumps
    for i in range(base_idx - 5, int(n * 0.3), -1):
        radius = min(search_radius, i, n - i - 1)
        if radius < 5:
            continue
        val = smoothed[i]
        if all(smoothed[i - off] >= val for off in range(1, radius + 1)) and \
           all(smoothed[i + off] >= val for off in range(1, radius + 1)):
            pastern_idx = i
            break
    if pastern_idx is None:
        pastern_idx = max(1, base_idx - int(base_idx * 0.25))

    # --- P2: Fetlock joint — widest first region from top (local maximum ABOVE pastern) ---
    fetlock_idx = None
    search_radius_fetlock = max(20, int(n * 0.1)) # ~10% of leg height ensures we find the massive fetlock bump
    for i in range(pastern_idx - int(n * 0.05), int(n * 0.1), -1):
        radius = min(search_radius_fetlock, i, n - i - 1)
        if radius < 5:
            continue
        val = smoothed[i]
        if all(smoothed[i - off] <= val for off in range(1, radius + 1)) and \
           all(smoothed[i + off] <= val for off in range(1, radius + 1)):
            fetlock_idx = i
            break
    if fetlock_idx is None:
        # Fallback if no clean local max is found: absolute max width in the middle 50%
        mid_start, mid_end = int(n * 0.3), pastern_idx
        if mid_end > mid_start:
            fetlock_idx = mid_start + int(np.argmax(smoothed[mid_start:mid_end]))
        else:
            fetlock_idx = max(0, pastern_idx - int(n * 0.15))

    # --- P3: Coronet band (Fallback) ---
    # The coronet band is the top of the hoof capsule, which is below the narrow pastern.
    coronet_idx = pastern_idx + int((base_idx - pastern_idx) * 0.40)

    # --- P1: Top of cannon bone ---
    # The cannon bone is the straight, uniform section between the knee and fetlock.
    # We trace UP using RAW widths to avoid moving-average smearing from top-mask artifacts.
    p1_idx = fetlock_idx
    if fetlock_idx > 10:
        base_cx = raw_cx[fetlock_idx]
        current_min_w = widths[fetlock_idx]
        
        for i in range(fetlock_idx - 1, 0, -1):
            # Stop if we hit extreme noise/gap
            if widths[i] < 10:
                break
            
            # Stop if the center line jumps completely off the leg (artifact)
            if abs(raw_cx[i] - base_cx) > current_min_w:
                break
                
            # Update the minimum width seen so far
            if widths[i] < current_min_w:
                current_min_w = widths[i]
                
            # Stop if the leg flares out by 35% (hitting the knee/carpus or chest)
            # Using 35% because raw widths fluctuate more than smoothed widths.
            if widths[i] > current_min_w * 1.35:
                break
                
            p1_idx = i
                
    p1 = (int(round(raw_cx[p1_idx])), int(y_coords[p1_idx]))
    p2 = (scx(fetlock_idx), int(y_coords[fetlock_idx]))
    p3 = (scx(coronet_idx), int(y_coords[coronet_idx]))
    p4 = (scx(base_idx),    int(y_coords[base_idx]))

    # ----------------------------------------------------------------
    # Hoof wall edge points (left & right) at coronet and base rows
    # Used to draw the hoof wall lines and measure medial/lateral angles
    # exactly like the reference image (Image 2).
    # We average a small band of rows around each key Y to be noise-robust.
    # ----------------------------------------------------------------
    def edge_at_row(target_y: int, band: int = 5) -> tuple[tuple[int,int], tuple[int,int]]:
        """Return (left_pt, right_pt) averaged over ±band rows around target_y."""
        lefts, rights = [], []
        for dy in range(-band, band + 1):
            ry = int(target_y) + dy
            if 0 <= ry < leg_mask.shape[0]:
                xs = np.where(leg_mask[ry] > 0)[0]
                if xs.size >= 2:
                    lefts.append(int(xs[0]))
                    rights.append(int(xs[-1]))
        if not lefts:
            # fallback: just use the center point
            cx = scx(coronet_idx)
            return (cx, int(target_y)), (cx, int(target_y))
        lx = int(round(float(np.mean(lefts))))
        rx = int(round(float(np.mean(rights))))
        return (lx, int(target_y)), (rx, int(target_y))

    p3_y = int(y_coords[coronet_idx])
    p4_y = int(y_coords[base_idx])
    p3_left, p3_right = edge_at_row(p3_y, band=6)
    p4_left, p4_right = edge_at_row(p4_y, band=6)

    # ----------------------------------------------------------------
    # Deviation metrics
    # ----------------------------------------------------------------
    # We define the plumb (ideal vertical) axis as the vertical line
    # passing through P1.  Any lateral shift of P2/P3/P4 from that
    # line is a real anatomical deviation.
    #
    # We also compute the angular deviation: the angle between the
    # P1→P2 cannon segment and a perfect vertical.  A straight leg
    # gives 0d; positive = leaning right, negative = leaning left.
    # ----------------------------------------------------------------
    plumb_x = float(p1[0])

    def lateral_dev(pt):
        """Lateral deviation from plumb (px). Positive = right."""
        return float(pt[0]) - plumb_x

    def segment_angle_from_vertical(pa, pb):
        """Angle (degrees) of segment pa→pb from vertical.
        Positive = leans right, negative = leans left."""
        dx = float(pb[0]) - float(pa[0])
        dy = float(pb[1]) - float(pa[1])
        if abs(dy) < 1e-6:
            return 0.0
        return float(np.degrees(np.arctan2(dx, dy)))   # atan2(Δx, Δy) from vertical

    cannon_angle   = segment_angle_from_vertical(p1, p2)   # P1→P2
    pastern_angle  = segment_angle_from_vertical(p2, p3)   # P2→P3
    hpa_dev_angle  = segment_angle_from_vertical(p3, p4)   # P3→P4

    # Lateral offsets at each key joint vs the cannon top (P1 as reference)
    p2_lat = lateral_dev(p2)
    p3_lat = lateral_dev(p3)
    p4_lat = lateral_dev(p4)

    logging.info(
        "Deviation — cannon angle=%.1fd | pastern angle=%.1fd | "
        "P2 offset=%+.1fpx | P3 offset=%+.1fpx | P4 offset=%+.1fpx",
        cannon_angle, pastern_angle, p2_lat, p3_lat, p4_lat
    )

    return {
        'p1_top':      p1,
        'p2_fetlock':  p2,
        'p3_coronet':  p3,
        'p4_hoof':     p4,
        # left/right edge points at coronet and hoof base (for wall-angle lines)
        'p3_left':     p3_left,
        'p3_right':    p3_right,
        'p4_left':     p4_left,
        'p4_right':    p4_right,
        # deviation metrics (used by draw_joint_overlay for on-image display)
        'cannon_angle_deg':  cannon_angle,
        'pastern_angle_deg': pastern_angle,
        'hpa_dev_angle_deg': hpa_dev_angle,
        'p2_lateral_px':     p2_lat,
        'p3_lateral_px':     p3_lat,
        'p4_lateral_px':     p4_lat,
    }


def draw_label_with_pointer(img: np.ndarray, text: str, pt: tuple[int, int], direction: str = "right"):
    """Draw a clean text label with a pointer line to pt, ensuring it stays on-screen."""
    h, w = img.shape[:2]
    # Font options
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Choose offset based on direction and compute text box bounds
    offset_x = int(w * 0.22) if direction == "right" else -int(w * 0.22)
    pointer_end = (pt[0] + offset_x, pt[1])
    
    if direction == "right":
        if pointer_end[0] + tw + 12 >= w:
            # Flip to left if it clips on the right
            pointer_end = (pt[0] - int(w * 0.22), pt[1])
            text_pt = (pointer_end[0] - tw - 6, pointer_end[1] + th // 2)
        else:
            text_pt = (pointer_end[0] + 6, pointer_end[1] + th // 2)
    else:
        if pointer_end[0] - tw - 12 < 0:
            # Flip to right if it clips on the left
            pointer_end = (pt[0] + int(w * 0.22), pt[1])
            text_pt = (pointer_end[0] + 6, pointer_end[1] + th // 2)
        else:
            text_pt = (pointer_end[0] - tw - 6, pointer_end[1] + th // 2)

    # Draw pointer circle and line
    cv2.circle(img, pt, 6, (0, 0, 255), -1)
    cv2.line(img, pt, pointer_end, (0, 0, 255), 2)
    
    # Draw background box for text
    bg_p1 = (text_pt[0] - 6, text_pt[1] - th - 6)
    bg_p2 = (text_pt[0] + tw + 6, text_pt[1] + baseline + 6)
    cv2.rectangle(img, bg_p1, bg_p2, (20, 20, 20), -1)
    cv2.rectangle(img, bg_p1, bg_p2, (0, 0, 255), 1)
    
    # Draw text
    cv2.putText(img, text, text_pt, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Anatomical joint overlay — P1 (top/cannon) → P2 (fetlock) → P3 (coronet)
# → P4 (hoof), with segmented dashed axes and joint angles.
# ---------------------------------------------------------------------------

def draw_dashed_line(img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple,
                     thickness: int = 2, dash_length: int = 10, gap_length: int = 8):
    """Draw a dashed line between pt1 and pt2."""
    p1 = np.array(pt1, dtype=np.float64)
    p2 = np.array(pt2, dtype=np.float64)
    dist = float(np.linalg.norm(p2 - p1))
    if dist < 1e-3:
        return
    direction = (p2 - p1) / dist
    step = dash_length + gap_length
    n_dashes = int(dist // step) + 1
    for i in range(n_dashes):
        start = p1 + direction * (i * step)
        if np.linalg.norm(start - p1) >= dist:
            break
        end = start + direction * dash_length
        if np.linalg.norm(end - p1) > dist:
            end = p2
        cv2.line(img,
                 (int(round(start[0])), int(round(start[1]))),
                 (int(round(end[0])), int(round(end[1]))),
                 color, thickness, cv2.LINE_AA)


def calculate_angle(a: tuple, vertex: tuple, b: tuple) -> float:
    """Interior angle (degrees) at `vertex`, between rays vertex→a and vertex→b."""
    v1 = np.array([a[0] - vertex[0], a[1] - vertex[1]], dtype=np.float64)
    v2 = np.array([b[0] - vertex[0], b[1] - vertex[1]], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def draw_joint_marker(img: np.ndarray, pt: tuple, label: str | None = None, radius: int = 7):
    """Draw a red dot with a white ring at a joint, with an optional small label."""
    pt = (int(round(pt[0])), int(round(pt[1])))
    cv2.circle(img, pt, radius + 3, (255, 255, 255), 2, cv2.LINE_AA)   # white ring
    cv2.circle(img, pt, radius, (40, 40, 235), -1, cv2.LINE_AA)        # red dot
    if label:
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        tx, ty = pt[0] + radius + 8, pt[1] - radius - 6
        bg_p1 = (tx - 4, ty - th - 4)
        bg_p2 = (tx + tw + 4, ty + baseline + 4)
        cv2.rectangle(img, bg_p1, bg_p2, (20, 20, 20), -1)
        cv2.rectangle(img, bg_p1, bg_p2, (255, 255, 255), 1)
        cv2.putText(img, label, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_angle_arc(img: np.ndarray, vertex: tuple, pt_a: tuple, pt_b: tuple,
                   color: tuple, radius: int = 30, label: str | None = None):
    """Draw a small arc at `vertex` spanning the angle between rays vertex→pt_a
    and vertex→pt_b, with an optional degree label placed at the arc midpoint."""
    vertex = (int(round(vertex[0])), int(round(vertex[1])))

    def ang_deg(p):
        return float(np.degrees(np.arctan2(p[1] - vertex[1], p[0] - vertex[0])))

    a1 = ang_deg(pt_a) % 360.0
    a2 = ang_deg(pt_b) % 360.0
    diff = (a2 - a1) % 360.0
    start_angle = a1
    if diff > 180.0:
        start_angle, diff = a2, 360.0 - diff

    cv2.ellipse(img, vertex, (radius, radius), 0, start_angle, start_angle + diff,
               color, 2, cv2.LINE_AA)

    if label:
        mid_angle = np.radians(start_angle + diff / 2.0)
        lx = int(vertex[0] + (radius + 24) * np.cos(mid_angle))
        ly = int(vertex[1] + (radius + 24) * np.sin(mid_angle))
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        bg_p1 = (lx - tw // 2 - 5, ly - th - 5)
        bg_p2 = (lx + tw // 2 + 5, ly + baseline + 5)
        cv2.rectangle(img, bg_p1, bg_p2, (20, 20, 20), -1)
        cv2.rectangle(img, bg_p1, bg_p2, color, 1)
        cv2.putText(img, label, (lx - tw // 2, ly), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_segment_label(img: np.ndarray, pt: tuple, text: str):
    """Draw a small floating text label (no pointer) centered at `pt` — used
    for naming a region between two joints, e.g. 'Cannon bone', 'Pastern'."""
    pt = (int(round(pt[0])), int(round(pt[1])))
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    tx, ty = pt[0] - tw // 2, pt[1]
    bg_p1 = (tx - 6, ty - th - 6)
    bg_p2 = (tx + tw + 6, ty + baseline + 6)
    cv2.rectangle(img, bg_p1, bg_p2, (20, 20, 20), -1)
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _angle_from_horizontal(pt_vertex: tuple, pt_other: tuple) -> float:
    """Angle of segment vertex→other measured FROM the horizontal (degrees).
    Returns value in [0, 180]. Used for hoof wall angles vs the ground line."""
    dx = float(pt_other[0]) - float(pt_vertex[0])
    dy = float(pt_other[1]) - float(pt_vertex[1])
    # angle from horizontal — arctan(|dy|/|dx|)
    if abs(dx) < 1e-6:
        return 90.0
    return float(np.degrees(np.arctan2(abs(dy), abs(dx))))


def draw_joint_overlay(img: np.ndarray, points: dict, dominant: str | None = None,
                       leg_index: int = 0) -> dict:
    """Draw the full anatomical joint overlay on `img` — Image-2 style.

    Anatomy drawn:
      ① Solid white vertical centerline  P1 → past P4 (ground)
      ② Dashed blue  P1→P2  (cannon bone segment)
      ③ Dashed green P2→P3  (pastern segment)
      ④ Fetlock angle arc + label at P2
      ⑤ Solid horizontal baseline at P3 spanning left→right coronet edge
      ⑥ Left hoof-wall line  P3_left  → P4_left   + medial angle label
      ⑦ Right hoof-wall line P3_right → P4_right  + lateral angle label
      ⑧ HPA arc at P3 center (angle between pastern axis and hoof centre)
      ⑨ Joint markers (X cross) at P1, P2, P3_left, P3_center, P3_right,
                                    P4_left, P4_center, P4_right
      ⑩ Deviation panel (bottom-left)

    `points` dict must contain:
        p1_top, p2_fetlock, p3_coronet, p4_hoof  — centre (x,y) points
        p3_left, p3_right                          — coronet left/right edges
        p4_left, p4_right                          — hoof base left/right edges
        + deviation keys from find_leg_landmarks
    """
    p1  = tuple(int(v) for v in points['p1_top'])
    p2  = tuple(int(v) for v in points['p2_fetlock'])
    p3  = tuple(int(v) for v in points['p3_coronet'])
    p4  = tuple(int(v) for v in points['p4_hoof'])
    p3L = tuple(int(v) for v in points.get('p3_left',  p3))
    p3R = tuple(int(v) for v in points.get('p3_right', p3))
    p4L = tuple(int(v) for v in points.get('p4_left',  p4))
    p4R = tuple(int(v) for v in points.get('p4_right', p4))

    h_img, w_img = img.shape[:2]

    COLOR_WHITE = (255, 255, 255)
    COLOR_BLUE  = (235, 130,  40)
    COLOR_GREEN = ( 90, 220,  90)   # plumb line
    COLOR_PINK  = (180, 100, 220)   # hoof wall lines
    COLOR_YELLOW= ( 50, 220, 255)   # cannon/pastern axis
    COLOR_RED   = (  0,   0, 255)   # joint dots

    # ① Solid green vertical plumb line: P1 → ground level
    ground_y = min(h_img - 1, p4[1] + 20)
    cv2.line(img, p1, (p1[0], ground_y), COLOR_GREEN, 2, cv2.LINE_AA)

    # ② Yellow cannon/pastern segment P1→P2→P4
    cv2.line(img, p1, p2, COLOR_YELLOW, 2, cv2.LINE_AA)
    cv2.line(img, p2, p4, COLOR_YELLOW, 2, cv2.LINE_AA)

    # ⑥ Left hoof wall line: p3_left → p4_left
    cv2.line(img, p3L, p4L, COLOR_PINK, 2, cv2.LINE_AA)

    # ⑦ Right hoof wall line: p3_right → p4_right
    cv2.line(img, p3R, p4R, COLOR_PINK, 2, cv2.LINE_AA)

    # Ground line through P4 (solid pink)
    margin = max(10, (p3R[0] - p3L[0]) // 4)
    cv2.line(img, (p4L[0] - margin, p4[1]), (p4R[0] + margin, p4[1]), COLOR_PINK, 2, cv2.LINE_AA)

    # Base Angle
    base_angle = calculate_angle(p4L, p4, p4R)
    draw_angle_arc(img, p4, p4L, p4R, COLOR_WHITE, radius=20, label=f"{base_angle:.0f}d")

    # ④ Fetlock angle (P1→P2→P4 interior angle)
    fetlock_angle = calculate_angle(p1, p2, p4)
    draw_angle_arc(img, p2, p1, p4, COLOR_WHITE, radius=32,
                   label=f"{fetlock_angle:.0f}d")

    # ⑧ HPA angle at P3 — we don't draw this for frontal, but keep the variable
    hpa_angle = calculate_angle(p2, p3, p4)

    # Hoof wall angles measured from the horizontal baseline
    left_wall_angle  = _angle_from_horizontal(p3L, p4L)
    right_wall_angle = _angle_from_horizontal(p3R, p4R)
    draw_angle_arc(img, p4L, p3L, (p4R[0] + margin, p4[1]), COLOR_PINK, radius=26,
                   label=f"{left_wall_angle:.0f}d")
    draw_angle_arc(img, p4R, (p4L[0] - margin, p4[1]), p3R, COLOR_PINK, radius=26,
                   label=f"{right_wall_angle:.0f}d")

    # ⑨ Joint markers (Red Dots matching Image 2)
    def draw_dot(pt, color=COLOR_RED, radius=5):
        pt = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)

    for pt in [p1, p2, p3L, p3, p3R, p4L, p4, p4R]:
        draw_dot(pt)

    # Physical Scale / Ruler at the bottom
    # Assume hoof width (P4_right - P4_left) is approx 13 cm.
    hoof_px_width = max(1, p4R[0] - p4L[0])
    px_per_cm = hoof_px_width / 13.0
    ruler_y = ground_y - 5
    ruler_start_x = p4L[0] - margin
    ruler_end_x = p4R[0] + margin
    cv2.line(img, (ruler_start_x, ruler_y), (ruler_end_x, ruler_y), COLOR_WHITE, 2, cv2.LINE_AA)

    for cm in range(0, 16, 3):
        x_tick = int(p4L[0] - int(1.5 * px_per_cm) + cm * px_per_cm)
        if x_tick > w_img - 10: break
        cv2.line(img, (x_tick, ruler_y - 5), (x_tick, ruler_y + 5), COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(img, f"{cm:.1f}", (x_tick - 10, ruler_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)

    # Deviation of hoof center from plumb line
    dev_cm = (p4[0] - p1[0]) / px_per_cm
    cv2.putText(img, f"{abs(dev_cm):.1f} cm", (p4[0] + 10, p4[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_RED, 1, cv2.LINE_AA)

    # ⑩ Deviation readout panel
    cannon_ang  = float(points.get('cannon_angle_deg',  0.0))
    pastern_ang = float(points.get('pastern_angle_deg', 0.0))
    p2_lat      = float(points.get('p2_lateral_px',     0.0))
    p4_lat      = float(points.get('p4_lateral_px',     0.0))

    def _side(val):
        if abs(val) < 2:   return "straight"
        return f"{'R' if val > 0 else 'L'} {abs(val):.0f}px"

    def _ang_side(val):
        if abs(val) < 0.3: return "0.0d"
        return f"{'R' if val > 0 else 'L'} {abs(val):.1f}d"

    dev_lines = [
        f"Cannon lean  : {_ang_side(cannon_ang)}",
        f"Pastern lean : {_ang_side(pastern_ang)}",
        f"Fetlock off  : {_side(p2_lat)}",
        f"Hoof off     : {_side(p4_lat)}",
        f"Wall L / R   : {left_wall_angle:.0f}d / {right_wall_angle:.0f}d",
        f"Fetlock angle: {fetlock_angle:.0f}d",
        f"HPA angle    : {hpa_angle:.0f}d",
    ]

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    line_h = 21
    total_h = len(dev_lines) * line_h + 12
    panel_y0 = h_img - total_h - 10 - leg_index * (total_h + 15)
    panel_x0 = 10
    max_tw = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in dev_lines)
    cv2.rectangle(img,
                  (panel_x0 - 4,  panel_y0 - 6),
                  (panel_x0 + max_tw + 8, panel_y0 + total_h),
                  (20, 20, 20), -1)
    cv2.rectangle(img,
                  (panel_x0 - 4,  panel_y0 - 6),
                  (panel_x0 + max_tw + 8, panel_y0 + total_h),
                  (100, 100, 100), 1)
    for k, line in enumerate(dev_lines):
        cy = panel_y0 + k * line_h + line_h - 2
        cv2.putText(img, line, (panel_x0, cy), font, scale, COLOR_WHITE, thick, cv2.LINE_AA)

    if dominant:
        cv2.putText(img, f"Leg {leg_index + 1} Dominant: {dominant}",
                    (10, 30 + leg_index * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return {
        "fetlock_angle":      fetlock_angle,
        "hpa_angle":          hpa_angle,
        "left_wall_angle":    left_wall_angle,
        "right_wall_angle":   right_wall_angle,
        "cannon_angle_deg":   cannon_ang,
        "pastern_angle_deg":  pastern_ang,
        "p2_lateral_px":      p2_lat,
        "p4_lateral_px":      p4_lat,
    }



def make_output_dir(base_dir) -> "Path":
    """Create and return results/<YYYYMMDD_HHMMSS>/ under base_dir."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


# --- V3 LOGIC START ---
def find_cannon_bone_axis(leg_mask: np.ndarray,
                          target_knee: tuple[float, float] | None = None,
                          target_hoof: tuple[float, float] | None = None
                          ) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return a strictly vertical centre-line axis for the cannon bone.

    FIX #8 (IMPLEMENTED): Uses the median X of cannon-zone row midpoints.
    Both pt_top and pt_bottom share the same X coordinate so the rendered
    line is always at 90d from the ground — no diagonal slant.

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

# --- V3 LOGIC END ---

def process_image(original_path: str, processed_path: str,
                  do_debug: bool = False, inferencer=None) -> None:
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

    # All outputs go into results/<timestamp>/ next to the original image
    out_dir = make_output_dir(p.parent)

    def save(filename: str, image) -> None:
        """Write image into the timestamped output directory."""
        cv2.imwrite(str(out_dir / filename), image)

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
    save(f"{p.stem}_depth.png", depth_color)
    logging.info("Saved depth map.")

    # --- Depth prefilter: remove far/back-leg pixels (FIX #9 / #12 / #13) ---
    depth_mask = depth_prefilter_mask(mask, depth_map, depth_delta=0.27)
    if do_debug:
        save(f"{p.stem}_depth_mask.png", depth_mask)
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
            for leg_info in legs:
                knee, hoof = leg_info['knee'], leg_info['hoof']
                line_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(line_mask,
                         (int(knee[0]), int(knee[1])),
                         (int(hoof[0]), int(hoof[1])), 255, thickness=5)
                overlap = (line_mask > 0) & (mask > 0)
                avg_d = float(depth_map[overlap].mean()) if np.any(overlap) else 0.0
                logging.info("Leg candidate knee=(%.1f,%.1f) hoof=(%.1f,%.1f) depth=%.3f",
                             *knee, *hoof, avg_d)
                if avg_d > best_depth_val:
                    best_depth_val, best_leg = avg_d, leg_info

            if best_leg is not None:
                knee, hoof = best_leg['knee'], best_leg['hoof']
                lm = select_front_leg_from_keypoints(depth_mask, knee, hoof, debug=do_debug)
                if lm is not None:
                    lm = trim_upper_leg_fraction(lm, 0.05)
                    leg_masks.append(lm)
                    info_dict = {'mask': lm, 'knee': knee, 'hoof': hoof}
                    if best_leg.get('type') == '4kp':
                        info_dict['ai_keypoints'] = best_leg['ai_keypoints']
                    leg_infos.append(info_dict)
                    save(f"{p.stem}_isolated_leg.png", lm)
                    logging.info("Saved isolated leg (AI path).")

    # --- Depth-based continuous-leg isolation ---
    # Tried before the legacy width-split fallback, since that fallback can
    # fragment the leg at the fetlock waist (see isolate_legs_from_depth
    # docstring) — this keeps the leg as one piece from cannon to hoof.
    if not leg_masks:
        depth_legs = isolate_legs_from_depth(depth_mask, depth_map, debug=do_debug)
        for lm in depth_legs:
            lm = trim_upper_leg_fraction(lm, 0.05)
            leg_masks.append(lm)
            
            info_dict = {'mask': lm, 'knee': None, 'hoof': None, 'is_hybrid': True}
            
            # --- Forced AI inference for Hybrid Approach ---
            if inferencer is not None:
                ys, xs = np.where(lm > 0)
                if ys.size > 0:
                    bbox = [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]
                    forced_legs = get_ai_leg_keypoints(inferencer, str(p), bboxes=[bbox])
                    if forced_legs:
                        forced_leg = forced_legs[0]
                        info_dict['knee'] = forced_leg['knee']
                        info_dict['hoof'] = forced_leg['hoof']
                        if forced_leg.get('type') == '4kp':
                            info_dict['ai_keypoints'] = forced_leg['ai_keypoints']
                            info_dict['ai_scores'] = forced_leg.get('ai_scores')
                        logging.info("Forced AI inference on depth mask successful.")
                        
            leg_infos.append(info_dict)
            
        if leg_masks:
            logging.info("Used depth-based continuous-leg isolation (%d leg(s)).", len(leg_masks))
            save(f"{p.stem}_isolated_leg.png", leg_masks[0])
            logging.info("Saved isolated leg (depth-isolation path).")

    # --- Fallback path ---
    if not leg_masks:
        logging.warning("AI keypoints missing or failed — using fallback leg selection.")
        lm = select_front_leg_fallback(depth_mask, depth_map=depth_map, debug=do_debug)
        if lm is None:
            logging.warning("No front leg found for %s", p.name)
            save(f"{p.stem}_analyzed.jpg", img)
            return
        lm = trim_upper_leg_fraction(lm, 0.20)
        leg_masks = [lm]
        leg_infos = [{'mask': lm, 'knee': (w / 2.0, 0.0), 'hoof': (w / 2.0, float(h - 1))}]
        save(f"{p.stem}_isolated_leg.png", lm)
        logging.info("Saved isolated leg (fallback path).")

    # --- Symmetry analysis (masks computed from nobg) ---
    combined_green = np.zeros((h, w), dtype=np.uint8)
    combined_red = np.zeros((h, w), dtype=np.uint8)
    per_leg_draw: list[dict] = []

    for info in leg_infos:
        pt_top, pt_bottom = find_cannon_bone_axis(
            info['mask'],
            target_knee=info.get('knee'),
            target_hoof=info.get('hoof'),
        )
        green, red, dominant = analyze_symmetry(info['mask'], pt_top, pt_bottom)
        combined_green = np.maximum(combined_green, green)
        combined_red = np.maximum(combined_red, red)

        # --- Anatomical landmarks via width-profile scan or AI ---
        is_hybrid = info.get('is_hybrid', False)
        if 'ai_keypoints' in info and info['ai_keypoints'] and not is_hybrid:
            ai_pts = info['ai_keypoints']
            landmarks = {
                'p1_top': (int(ai_pts[0][0]), int(ai_pts[0][1])),
                'p2_fetlock': (int(ai_pts[1][0]), int(ai_pts[1][1])),
                'p3_coronet': (int(ai_pts[2][0]), int(ai_pts[2][1])),
                'p4_hoof': (int(ai_pts[3][0]), int(ai_pts[3][1]))
            }
            logging.info("Using explicit 4-keypoint AI predictions for landmarks (Front-Edge Strategy).")
        else:
            # Finds the real fetlock narrowing and the real coronet/hoof-flare
            # point, instead of approximating them as plain midpoints.
            landmarks = find_leg_landmarks(info['mask'])
            
            # --- Hybrid P3 Injection ---
            if is_hybrid and landmarks and 'ai_keypoints' in info and 'ai_scores' in info:
                ai_pts = info['ai_keypoints']
                ai_scores = info['ai_scores']
                p3_score = ai_scores[2]
                p4_score = ai_scores[3]
                
                if p3_score > 0.15 and p4_score > 0.15:
                    ai_hoof_height = ai_pts[3][1] - ai_pts[2][1]
                    if ai_hoof_height > 10:
                        hybrid_p3_y = landmarks['p4_hoof'][1] - ai_hoof_height
                        # Ensure it's between fetlock and hoof base
                        if landmarks['p2_fetlock'][1] < hybrid_p3_y < landmarks['p4_hoof'][1]:
                            # Find the center X of the mask at this new Y
                            ys, xs = np.where(info['mask'] > 0)
                            row_xs = xs[ys == int(hybrid_p3_y)]
                            if row_xs.size > 0:
                                hybrid_cx = int(round((row_xs[0] + row_xs[-1]) / 2.0))
                                old_p3 = landmarks['p3_coronet']
                                landmarks['p3_coronet'] = (hybrid_cx, int(hybrid_p3_y))
                                logging.info("Hybrid P3 applied! AI Hoof Height: %.1fpx. Replaced %s with %s", 
                                             ai_hoof_height, old_p3, landmarks['p3_coronet'])
                            else:
                                logging.warning("Hybrid P3 row out of mask bounds, falling back to math P3.")
                        else:
                            logging.warning("Hybrid P3_y (%.1f) not between fetlock and hoof base. Falling back.", hybrid_p3_y)
                else:
                    logging.info("AI confidence too low for Hybrid P3 (P3:%.2f, P4:%.2f). Using math P3.", p3_score, p4_score)
        if landmarks is None:
            logging.warning("Could not measure landmarks for a leg in %s — skipping its overlay.", p.name)
            continue

        logging.info("Segmented parts for %s:", p.name)
        logging.info("  - P1 Top of cannon bone: %s", landmarks['p1_top'])
        logging.info("  - P2 Fetlock joint: %s", landmarks['p2_fetlock'])
        logging.info("  - P3 Coronet band: %s", landmarks['p3_coronet'])
        logging.info("  - P4 Hoof base: %s", landmarks['p4_hoof'])

        per_leg_draw.append({
            'pt_top': pt_top,
            'pt_bottom': pt_bottom,
            'dominant': dominant,
            **landmarks,
        })

    # --- Final output: overlay is applied on the ORIGINAL image ---
    out = apply_overlay(img, combined_green, combined_red, alpha=0.55)

    for i, di in enumerate(per_leg_draw):
        angles = draw_joint_overlay(out, di, dominant=di['dominant'], leg_index=i)
        logging.info("Leg %d angles — Fetlock: %.1f deg, HPA: %.1f deg",
                     i + 1, angles['fetlock_angle'], angles['hpa_angle'])

    save(f"{p.stem}_analyzed.jpg", out)

    # --- Same landmarks, drawn on the depth map itself (Image-3 style) ---
    # Pixel coordinates are 1:1 with the original since depth_map/depth_color
    # were resized to match the original image dimensions earlier.
    depth_annotated = depth_color.copy()
    for i, di in enumerate(per_leg_draw):
        draw_joint_overlay(depth_annotated, di, dominant=di['dominant'], leg_index=i)
    save(f"{p.stem}_depth_annotated.png", depth_annotated)
    logging.info("Saved %s_depth_annotated.png", p.stem)

    if do_debug:
        dbg = img.copy()
        dbg[combined_green > 0] = [34, 197, 94]
        dbg[combined_red > 0] = [48, 48, 220]
        for i, di in enumerate(per_leg_draw):
            draw_joint_overlay(dbg, di, dominant=di['dominant'], leg_index=i)
        save(f"{p.stem}_debug.png", dbg)

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
    parser.add_argument("--model-config", type=str, default=None,
                        help="Optional local model config path (.py) for MMPoseInferencer")
    parser.add_argument("--model-weights", type=str, default=None,
                        help="Optional local model weights path (.pth) for MMPoseInferencer")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for MMPose (e.g. cpu or cuda:0)")
    args = parser.parse_args()

    inferencer = None
    if args.use_ai and MMPoseInferencer is not None:
        try:
            kwargs = {"device": args.device} if args.device else {}
            if args.model_weights:
                kwargs['pose2d_weights'] = args.model_weights
            pose = args.model_config or 'rtmpose-m_8xb64-210e_ap10k-256x256'
            inferencer = MMPoseInferencer(pose2d=pose, **kwargs)
            logging.info("MMPose inferencer initialized.")
        except Exception as e:
            logging.warning("Failed to initialize MMPoseInferencer: %s", e)

    try:
        process_image(args.original_image, args.processed_image, do_debug=args.debug, inferencer=inferencer)
    except Exception as e:
        logging.exception("Failed processing %s: %s", args.original_image, e)


if __name__ == "__main__":
    main()