"""
API v3 — Dual-model (MMPose + YOLO) clinical analysis.

Key additions over v2:
  - Both MMPose and YOLO run in parallel per leg.
  - Granular angles (HoofAngle, PasternAngle, AngleDeviation) per model.
  - Side-by-side comparison logging.
  - Full clinical parity: YOLO rejects bad results exactly like MMPose.
"""

from fastapi import APIRouter, Request, HTTPException
from apis.v3.schemas import AdvancedScanRequest, AdvancedScanResponse, ModelResult
from apis.v2.services.inference import get_image_bytes, run_leg_inference  # , run_yolo_inference
from apis.v2.services.scoring import calculate_leg_score
from apis.v2.services.quality import map_quality
from apis.v2.services.aggregator import aggregate_scan
from apis.v2.services.clinical import map_condition, map_clinical_notes, map_recommendation
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# Helper: Convert a raw predict() result into flat per-leg fields
# ─────────────────────────────────────────────────────────────

def _build_leg_payload(prediction: dict, leg_key: str) -> tuple[dict, float | None]:
    """
    Converts a raw predict() dict into a flat dict of per-leg API fields.
    Returns (payload_dict, score_or_None).

    Handles both success and failure cases identically for MMPose and YOLO.
    """
    payload = {}

    if not prediction.get("success"):
        # Failure path — mirrors MMPose rejection behaviour
        error_msg = prediction.get("error", "Unknown inference error. Please retake the image.")
        payload[f"{leg_key}ScanScore"]        = None
        payload[f"{leg_key}Notes"]            = error_msg
        payload[f"{leg_key}Condition"]        = None
        payload[f"{leg_key}Recommendation"]   = None
        payload[f"{leg_key}Quality"]          = None
        payload[f"{leg_key}QualityCheck"]     = "Fail"
        # Granular angles — all None on failure
        payload[f"{leg_key}HoofAngle"]        = None
        payload[f"{leg_key}PasternAngle"]     = None
        payload[f"{leg_key}AngleDeviation"]   = None
        return payload, None

    p_angle = prediction["pastern_angle"]
    h_angle = prediction["hoof_angle"]
    hpa_dev = prediction.get("hpa_dev")
    conf    = prediction.get("model_confidence") or 0.0

    score          = calculate_leg_score(p_angle, h_angle)
    quality        = map_quality(conf)
    notes          = map_clinical_notes(score)
    condition      = map_condition(score)
    recommendation = map_recommendation(score)

    payload[f"{leg_key}ScanScore"]        = score
    payload[f"{leg_key}Notes"]            = notes
    payload[f"{leg_key}Condition"]        = condition
    payload[f"{leg_key}Recommendation"]   = recommendation
    payload[f"{leg_key}Quality"]          = quality
    payload[f"{leg_key}QualityCheck"]     = "Pass"
    # Granular angles
    payload[f"{leg_key}HoofAngle"]        = h_angle
    payload[f"{leg_key}PasternAngle"]     = p_angle
    payload[f"{leg_key}AngleDeviation"]   = hpa_dev

    return payload, score


# ─────────────────────────────────────────────────────────────
# V3 Endpoint
# ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AdvancedScanResponse)
async def analyze_v3(request: AdvancedScanRequest, req: Request):
    predictor      = req.app.state.predictor
    # yolo_predictor = getattr(req.app.state, "yolo_predictor", None)

    if not predictor:
        raise HTTPException(status_code=503, detail="MMPose model not initialized")
    # if not yolo_predictor:
    #     raise HTTPException(status_code=503, detail="YOLO model not initialized")

    legs = {
        "frontLeft":  request.frontLeftLateral,
        "frontRight": request.frontRightLateral,
        "backLeft":   request.backLeftLateral,
        "backRight":  request.backRightLateral,
    }

    # ── 1. Download all images concurrently ──────────────────
    leg_keys    = []
    fetch_tasks = []
    for leg_key, image_input in legs.items():
        if image_input:
            leg_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(image_input))

    print(f"📡 [v3] Downloading {len(fetch_tasks)} images...")
    downloaded = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    leg_data: dict[str, bytes] = {}
    for i, leg_key in enumerate(leg_keys):
        if isinstance(downloaded[i], Exception):
            print(f"❌ [v3] Download failed for {leg_key}: {downloaded[i]}")
        else:
            leg_data[leg_key] = downloaded[i]

    # ── 2. Dual-model inference inside ThreadPoolExecutor ────
    # Each thread handles one leg: runs MMPose then YOLO sequentially.
    # This keeps GPU resource contention low while freeing the async loop.

    def process_single_leg(leg_key: str, img_bytes: bytes):
        mp = run_leg_inference(predictor, img_bytes)
        # yl = run_yolo_inference(yolo_predictor, img_bytes)
        return leg_key, mp, {}  # yl disabled

    print(f"🧠 [v3] MMPose-only inference on {len(leg_data)} legs...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [
            loop.run_in_executor(pool, process_single_leg, k, b)
            for k, b in leg_data.items()
        ]
        inference_results = await asyncio.gather(*tasks)

    # ── 3. Build MMPose field dict ───────────────────────────
    mmpose_fields: dict = {}
    mmpose_scores: list = []

    for leg_key, mp_pred, _ in inference_results:
        mp_payload, mp_score = _build_leg_payload(mp_pred, leg_key)
        mmpose_fields.update(mp_payload)
        
        if mp_score is not None:
            mmpose_scores.append(mp_score)

        # Clinical log
        print(f"📊 [v3] {leg_key}: MMPose → P={mp_pred.get('pastern_angle')}°  H={mp_pred.get('hoof_angle')}°  Dev={mp_pred.get('hpa_dev')}°  Score={mp_score}")

    # ── 4. Fill missing legs (no image supplied) with None ───
    for leg_key, image_input in legs.items():
        if not image_input:
            for suffix in (
                "ScanScore", "Notes", "Condition", "Recommendation",
                "Quality", "QualityCheck",
                "HoofAngle", "PasternAngle", "AngleDeviation"
            ):
                mmpose_fields.setdefault(f"{leg_key}{suffix}", None)

    # ── 5. Aggregate (MMPose as primary source) ──────────────
    aggregation = aggregate_scan(mmpose_scores)

    return AdvancedScanResponse(
        mmpose=ModelResult(**mmpose_fields),
        yolo=None,  # YOLO disabled per project requirements
        scanScore=aggregation["scanScore"],
        notes=aggregation["notes"],
        quality=1,
    )
