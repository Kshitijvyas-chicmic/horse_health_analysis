"""
API v4 — Same clinical flow as v3, for mobile pre-cutout images.

- No server-side background removal (rembg not loaded at startup).
- Overall scanScore uses lateral slots only (same as v3).
"""

import logging
from fastapi import APIRouter, Request, HTTPException
from apis.v5.schemas import AdvancedScanRequest, AdvancedScanResponseV5, ModelResultV5
from apis.v5.services.inference import get_image_bytes, run_leg_inference, process_frontal_leg_symmetry, process_lateral_leg_overlay
from apis.v2.services.scoring import calculate_leg_score
from apis.v2.services.quality import map_quality
from apis.v2.services.aggregator import aggregate_scan
from apis.v2.services.clinical import map_condition, map_clinical_notes, map_recommendation
import asyncio
import gc
import hashlib
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

_LATERAL_KEYS_FOR_TOP_LEVEL_SCORE = frozenset(
    {"frontLeft", "frontRight", "backLeft", "backRight"}
)


def _build_leg_payload(prediction: dict, leg_key: str) -> tuple[dict, float | None]:
    payload = {}

    if not prediction.get("success"):
        error_msg = prediction.get("error", "Unknown inference error. Please retake the image.")
        payload[f"{leg_key}ScanScore"] = 0.0
        payload[f"{leg_key}Notes"] = error_msg
        payload[f"{leg_key}Condition"] = None
        payload[f"{leg_key}Recommendation"] = None
        payload[f"{leg_key}Quality"] = 0.0
        payload[f"{leg_key}QualityCheck"] = "Fail"
        payload[f"{leg_key}HoofAngle"] = 0.0
        payload[f"{leg_key}PasternAngle"] = 0.0
        payload[f"{leg_key}AngleDeviation"] = 0.0
        return payload, None

    p_angle = prediction["pastern_angle"]
    h_angle = prediction["hoof_angle"]
    hpa_dev = prediction.get("hpa_dev")
    conf = prediction.get("model_confidence") or 0.0

    score = calculate_leg_score(p_angle, h_angle)
    quality = map_quality(conf)
    notes = map_clinical_notes(score)
    condition = map_condition(score)
    recommendation = map_recommendation(score)

    payload[f"{leg_key}ScanScore"] = score
    payload[f"{leg_key}Notes"] = notes
    payload[f"{leg_key}Condition"] = condition
    payload[f"{leg_key}Recommendation"] = recommendation
    payload[f"{leg_key}Quality"] = quality
    payload[f"{leg_key}QualityCheck"] = "Pass"
    payload[f"{leg_key}HoofAngle"] = h_angle
    payload[f"{leg_key}PasternAngle"] = p_angle
    payload[f"{leg_key}AngleDeviation"] = hpa_dev

    return payload, score


@router.post("/analyze", response_model=AdvancedScanResponseV5)
async def analyze_v5(request: AdvancedScanRequest, req: Request):
    predictor = req.app.state.predictor

    if not predictor:
        raise HTTPException(status_code=503, detail="MMPose model not initialized")

    # --- Log incoming request body summary ---
    def _has(val): return "✅" if val else "❌"
    logging.info(
        f"📥 [v5] Incoming request body:\n"
        f"  Lateral (original)  : frontLeft={_has(request.frontLeftLateral)}  frontRight={_has(request.frontRightLateral)}  backLeft={_has(request.backLeftLateral)}  backRight={_has(request.backRightLateral)}\n"
        f"  Lateral (processed) : frontLeft={_has(request.frontLeftLateralProcessed)}  frontRight={_has(request.frontRightLateralProcessed)}  backLeft={_has(request.backLeftLateralProcessed)}  backRight={_has(request.backRightLateralProcessed)}\n"
        f"  Frontal (original)  : frontLeft={_has(request.frontLeftFrontal)}  frontRight={_has(request.frontRightFrontal)}  backLeft={_has(request.backLeftFrontal)}  backRight={_has(request.backRightFrontal)}\n"
        f"  Frontal (processed) : frontLeft={_has(request.frontLeftFrontalProcessed)}  frontRight={_has(request.frontRightFrontalProcessed)}  backLeft={_has(request.backLeftFrontalProcessed)}  backRight={_has(request.backRightFrontalProcessed)}"
    )

    # --- Map lateral pairs: original + processed (bg-removed) ---
    # If only one is provided, the image is treated as the processed one (backward-compat).
    lateral_pairs = {
        "frontLeft":  (request.frontLeftLateral,  request.frontLeftLateralProcessed),
        "frontRight": (request.frontRightLateral, request.frontRightLateralProcessed),
        "backLeft":   (request.backLeftLateral,   request.backLeftLateralProcessed),
        "backRight":  (request.backRightLateral,  request.backRightLateralProcessed),
    }

    frontal_pairs = {
        "frontLeftFrontal": (request.frontLeftFrontal, request.frontLeftFrontalProcessed),
        "frontRightFrontal": (request.frontRightFrontal, request.frontRightFrontalProcessed),
        "backLeftFrontal": (request.backLeftFrontal, request.backLeftFrontalProcessed),
        "backRightFrontal": (request.backRightFrontal, request.backRightFrontalProcessed),
    }

    # Gather tasks for download
    fetch_tasks = []

    lateral_keys = []
    # Track whether each lateral slot has an overlay pair or just a single processed image
    lateral_has_overlay = {}
    for leg_key, (img_orig, img_proc) in lateral_pairs.items():
        if img_orig and img_proc:
            # Both provided: original for drawing, processed for inference
            lateral_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(img_orig))
            fetch_tasks.append(get_image_bytes(img_proc))
            lateral_has_overlay[leg_key] = True
        elif img_proc:
            # Only processed: run inference on it, no overlay
            lateral_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(img_proc))
            lateral_has_overlay[leg_key] = False
        elif img_orig:
            # Only original provided (backward-compat with old clients)
            lateral_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(img_orig))
            lateral_has_overlay[leg_key] = False

    frontal_keys = []
    for leg_key, (img_orig, img_proc) in frontal_pairs.items():
        if img_orig and img_proc:
            # Both provided (ideal case)
            frontal_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(img_orig))
            fetch_tasks.append(get_image_bytes(img_proc))
        elif img_orig or img_proc:
            # Only one provided — use the same image for both slots.
            # process_frontal_leg_symmetry can still run geometry analysis.
            single = img_orig or img_proc
            frontal_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(single))
            fetch_tasks.append(get_image_bytes(single))

    logging.info(f"📡 [v5] Downloading {len(fetch_tasks)} images...")
    downloaded = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Reconstruct lateral_data mapping
    lateral_data = {}  # leg_key -> bytes or (bytes_orig, bytes_proc)
    idx = 0
    for leg_key in lateral_keys:
        has_overlay = lateral_has_overlay[leg_key]
        if has_overlay:
            res_orig = downloaded[idx]
            res_proc = downloaded[idx + 1]
            idx += 2
            if isinstance(res_orig, Exception) or isinstance(res_proc, Exception):
                logging.error(f"❌ [v5] Download failed for lateral {leg_key}")
            else:
                lateral_data[leg_key] = (res_orig, res_proc)
        else:
            res = downloaded[idx]
            idx += 1
            if isinstance(res, Exception):
                logging.error(f"❌ [v5] Download failed for lateral {leg_key}: {res}")
            else:
                lateral_data[leg_key] = res  # single bytes

    frontal_data = {}
    for leg_key in frontal_keys:
        res_orig = downloaded[idx]
        res_proc = downloaded[idx + 1]
        idx += 2
        if isinstance(res_orig, Exception) or isinstance(res_proc, Exception):
            logging.error(f"❌ [v5] Download failed for {leg_key}")
        else:
            frontal_data[leg_key] = (res_orig, res_proc)

    def process_lateral_leg(leg_key: str, img_data):
        """Handles both single-image and overlay-pair lateral inference."""
        try:
            if isinstance(img_data, tuple):
                # Overlay mode: original + processed
                img_orig_bytes, img_proc_bytes = img_data
                mp, url = process_lateral_leg_overlay(predictor, img_orig_bytes, img_proc_bytes)
            else:
                # Single-image mode: run inference, then upload the annotated image from image_base64.
                # image_base64 is always populated by HPAPredictor.predict(), so we can always
                # get an output image even without a separate original image.
                from apis.v5.services.upload import upload_image_to_s3
                import base64 as _b64
                mp = run_leg_inference(predictor, img_data)
                url = ""
                output_b64 = mp.get("image_base64")
                if output_b64:
                    try:
                        analyzed_bytes = _b64.b64decode(output_b64)
                        url = upload_image_to_s3(analyzed_bytes, file_extension="jpg", folder="lateral_overlays") or ""
                    except Exception as upload_err:
                        logging.error(f"❌ [v5] {leg_key}: S3 upload failed in single-image mode: {upload_err}")
        except Exception as e:
            logging.error(f"❌ [v5] Lateral inference failed for {leg_key}: {e}")
            mp = {"success": False, "error": "We couldn't analyze this image. Please ensure the photo is clear and taken from the correct angle."}
            url = None
        return leg_key, mp, url

    def process_frontal_leg(leg_key: str, img_orig: bytes, img_proc: bytes):
        try:
            url = process_frontal_leg_symmetry(img_orig, img_proc)
            err_msg = None
        except Exception as e:
            url = None
            err_msg = "We couldn't analyze the symmetry. Please ensure the photo is clear and try again."
            logging.error(f"❌ [v5] Frontal inference failed for {leg_key}: {e}")
        return leg_key, err_msg, url

    logging.info(f"🧠 [v5] Processing {len(lateral_data)} lateral and {len(frontal_data)} frontal slot(s)...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        tasks = []
        for k, img_data in lateral_data.items():
            tasks.append(loop.run_in_executor(pool, process_lateral_leg, k, img_data))
        for k, (b_orig, b_proc) in frontal_data.items():
            tasks.append(loop.run_in_executor(pool, process_frontal_leg, k, b_orig, b_proc))
            
        inference_results = await asyncio.gather(*tasks)

    mmpose_fields: dict = {}
    mmpose_scores: list = []

    for leg_key, payload, url in inference_results:
        if "Frontal" in leg_key:
            mmpose_fields[f"{leg_key}ImageUrl"] = url
            for suffix in ("ScanScore", "Quality", "HoofAngle", "PasternAngle", "AngleDeviation"):
                mmpose_fields[f"{leg_key}{suffix}"] = 0.0
            mmpose_fields[f"{leg_key}Notes"] = payload if isinstance(payload, str) else None
            # QualityCheck for frontal is null on success (no angle scoring is done for frontal,
            # so 'Pass' with Score:0 is semantically wrong). Only set 'Fail' when an error occurred.
            mmpose_fields[f"{leg_key}QualityCheck"] = "Fail" if payload else None
            for suffix in ("Condition", "Recommendation"):
                mmpose_fields[f"{leg_key}{suffix}"] = None
            logging.info(f"📊 [v5] {leg_key}: Symmetry Analyzed, URL={url}")
        else:
            mp_pred = payload
            mp_payload, mp_score = _build_leg_payload(mp_pred, leg_key)
            mmpose_fields.update(mp_payload)
            # Store the lateral annotated image URL if one was returned
            if url:
                mmpose_fields[f"{leg_key}ImageUrl"] = url
            else:
                # Log explicitly when inference succeeded but no overlay image was produced.
                # This catches silent S3 upload failures or missing image_base64 from predict().
                if mp_pred.get("success"):
                    logging.warning(f"⚠️ [v5] {leg_key}: Inference succeeded (score computed) but ImageUrl is empty — S3 upload may have failed.")

            if mp_score is not None and leg_key in _LATERAL_KEYS_FOR_TOP_LEVEL_SCORE:
                mmpose_scores.append(mp_score)

            if mp_pred.get("success") is False:
                logging.error(f"❌ [v5] {leg_key} Lateral Inference Failed: {mp_pred.get('error', 'Unknown Error')}")
            else:
                logging.info(
                    f"📊 [v5] {leg_key}: P={mp_pred.get('pastern_angle')}° "
                    f"H={mp_pred.get('hoof_angle')}° Dev={mp_pred.get('hpa_dev')}° Score={mp_score} URL={url}"
                )

    # Set nulls for missing lateral inputs
    for leg_key, (img_orig, img_proc) in lateral_pairs.items():
        if not (img_orig or img_proc):
            err_msg = "No lateral image provided. Please upload an image."
            mp_payload, _ = _build_leg_payload({"success": False, "error": err_msg}, leg_key)
            mmpose_fields.update(mp_payload)
            
    for leg_key, (img_orig, img_proc) in frontal_pairs.items():
        if not (img_orig and img_proc):
            mmpose_fields[f"{leg_key}ImageUrl"] = None
            for suffix in ("ScanScore", "Quality", "HoofAngle", "PasternAngle", "AngleDeviation"):
                mmpose_fields[f"{leg_key}{suffix}"] = 0.0
            for suffix in ("Notes", "Condition", "Recommendation", "QualityCheck"):
                mmpose_fields[f"{leg_key}{suffix}"] = None

    aggregation = aggregate_scan(mmpose_scores)

    del lateral_data
    del frontal_data
    del downloaded
    gc.collect()

    return AdvancedScanResponseV5(
        mmpose=ModelResultV5(**mmpose_fields),
        yolo=None,
        scanScore=aggregation["scanScore"],
        notes=aggregation["notes"],
        quality=1,
    )
