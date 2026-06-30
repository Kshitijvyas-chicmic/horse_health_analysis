"""
API v4 — Same clinical flow as v3, for mobile pre-cutout images.

- No server-side background removal (rembg not loaded at startup).
- Overall scanScore uses lateral slots only (same as v3).
"""

from fastapi import APIRouter, Request, HTTPException
from apis.v5.schemas import AdvancedScanRequest, AdvancedScanResponse, ModelResult
from apis.v5.services.inference import get_image_bytes, run_leg_inference, process_frontal_leg_symmetry
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


@router.post("/analyze", response_model=AdvancedScanResponse)
async def analyze_v5(request: AdvancedScanRequest, req: Request):
    predictor = req.app.state.predictor

    if not predictor:
        raise HTTPException(status_code=503, detail="MMPose model not initialized")

    lateral_legs = {
        "frontLeft": request.frontLeftLateral,
        "frontRight": request.frontRightLateral,
        "backLeft": request.backLeftLateral,
        "backRight": request.backRightLateral,
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
    for leg_key, image_input in lateral_legs.items():
        if image_input:
            lateral_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(image_input))
            
    frontal_keys = []
    for leg_key, (img_orig, img_proc) in frontal_pairs.items():
        if img_orig and img_proc:
            frontal_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(img_orig))
            fetch_tasks.append(get_image_bytes(img_proc))

    print(f"📡 [v5] Downloading {len(fetch_tasks)} images...")
    downloaded = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Reconstruct leg_data mapping
    lateral_data = {}
    idx = 0
    for leg_key in lateral_keys:
        res = downloaded[idx]
        idx += 1
        if isinstance(res, Exception):
            print(f"❌ [v5] Download failed for {leg_key}: {res}")
        else:
            lateral_data[leg_key] = res
            
    frontal_data = {}
    for leg_key in frontal_keys:
        res_orig = downloaded[idx]
        res_proc = downloaded[idx + 1]
        idx += 2
        if isinstance(res_orig, Exception) or isinstance(res_proc, Exception):
            print(f"❌ [v5] Download failed for {leg_key}")
        else:
            frontal_data[leg_key] = (res_orig, res_proc)

    def process_lateral_leg(leg_key: str, img_bytes: bytes):
        try:
            mp = run_leg_inference(predictor, img_bytes)
        except Exception as e:
            print(f"❌ [v5] Lateral inference failed for {leg_key}: {e}")
            mp = {"success": False, "error": "We couldn't analyze this image. Please ensure the photo is clear and taken from the correct angle."}
        return leg_key, mp, None

    def process_frontal_leg(leg_key: str, img_orig: bytes, img_proc: bytes):
        try:
            url = process_frontal_leg_symmetry(img_orig, img_proc)
            err_msg = None
        except Exception as e:
            url = None
            err_msg = "We couldn't analyze the symmetry. Please ensure the photo is clear and try again."
            print(f"❌ [v5] Frontal inference failed for {leg_key}: {e}")
        return leg_key, err_msg, url

    print(f"🧠 [v5] Processing {len(lateral_data)} lateral and {len(frontal_data)} frontal slot(s)...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = []
        for k, b in lateral_data.items():
            tasks.append(loop.run_in_executor(pool, process_lateral_leg, k, b))
        for k, (b_orig, b_proc) in frontal_data.items():
            tasks.append(loop.run_in_executor(pool, process_frontal_leg, k, b_orig, b_proc))
            
        inference_results = await asyncio.gather(*tasks)

    mmpose_fields: dict = {}
    mmpose_scores: list = []

    for leg_key, payload, frontal_url in inference_results:
        if "Frontal" in leg_key:
            mmpose_fields[f"{leg_key}ImageUrl"] = frontal_url
            for suffix in ("ScanScore", "Quality", "HoofAngle", "PasternAngle", "AngleDeviation"):
                mmpose_fields[f"{leg_key}{suffix}"] = 0.0
                
            mmpose_fields[f"{leg_key}Notes"] = payload if isinstance(payload, str) else None
            mmpose_fields[f"{leg_key}QualityCheck"] = "Fail" if payload else None
            
            for suffix in ("Condition", "Recommendation"):
                mmpose_fields[f"{leg_key}{suffix}"] = None
                
            print(f"📊 [v5] {leg_key}: Symmetry Analyzed, URL={frontal_url}")
        else:
            mp_pred = payload
            mp_payload, mp_score = _build_leg_payload(mp_pred, leg_key)
            mmpose_fields.update(mp_payload)
    
            if mp_score is not None and leg_key in _LATERAL_KEYS_FOR_TOP_LEVEL_SCORE:
                mmpose_scores.append(mp_score)
    
            print(
                f"📊 [v5] {leg_key}: P={mp_pred.get('pastern_angle')}° "
                f"H={mp_pred.get('hoof_angle')}° Dev={mp_pred.get('hpa_dev')}° Score={mp_score}"
            )

    # Set nulls for missing inputs
    for leg_key, image_input in lateral_legs.items():
        if not image_input:
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

    return AdvancedScanResponse(
        mmpose=ModelResult(**mmpose_fields),
        yolo=None,
        scanScore=aggregation["scanScore"],
        notes=aggregation["notes"],
        quality=1,
    )
