from fastapi import APIRouter, Request, HTTPException
from apis.v2.schemas import AdvancedScanRequest, AdvancedScanResponse
from apis.v2.services.inference import get_image_bytes, run_leg_inference
from apis.v2.services.scoring import calculate_leg_score
from apis.v2.services.notes import map_leg_notes
from apis.v2.services.quality import map_quality
from apis.v2.services.aggregator import aggregate_scan
import traceback

router = APIRouter()

@router.post("/analyze", response_model=AdvancedScanResponse)
async def analyze_v2(request: AdvancedScanRequest, req: Request):
    predictor = req.app.state.predictor
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not initialized")

    legs = {
        "frontLeft": request.frontLeftLateral,
        "frontRight": request.frontRightLateral,
        "backLeft": request.backLeftLateral,
        "backRight": request.backRightLateral
    }

    # 1. Fetch all images concurrently
    import asyncio
    
    results = {}
    leg_keys = []
    fetch_tasks = []
    
    for leg_key, image_input in legs.items():
        if image_input:
            leg_keys.append(leg_key)
            fetch_tasks.append(get_image_bytes(image_input))
        else:
            results[f"{leg_key}ScanScore"] = None
            results[f"{leg_key}Notes"] = None
            results[f"{leg_key}Quality"] = None

    # Run downloads in parallel
    print(f"📡 Downloading {len(fetch_tasks)} images concurrently...")
    downloaded_images = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    
    # Map downloaded bytes back to leg keys for inference
    leg_data = {}
    for i, leg_key in enumerate(leg_keys):
        result = downloaded_images[i]
        if isinstance(result, Exception):
            print(f"❌ Download failed for {leg_key}: {result}")
            results[f"{leg_key}ScanScore"] = None
            results[f"{leg_key}Notes"] = f"Download failed: {str(result)}"
            results[f"{leg_key}Quality"] = None
        else:
            leg_data[leg_key] = result

    # 2. Sequential Inference (CPU intensive, so keep sequential to avoid thrashing)
    leg_scores = []
    for leg_key, img_bytes in leg_data.items():
        try:
            prediction = run_leg_inference(predictor, img_bytes)
            
            if not prediction.get("success"):
                results[f"{leg_key}ScanScore"] = None
                results[f"{leg_key}Notes"] = f"Inference failed: {prediction.get('error')}"
                results[f"{leg_key}Quality"] = None
                continue

            # 3. Calculate Leg Score
            p_angle = prediction["pastern_angle"]
            h_angle = prediction["hoof_angle"]
            conf = prediction["model_confidence"]
            
            score = calculate_leg_score(p_angle, h_angle)
            quality = map_quality(conf)
            notes = map_leg_notes(score)
            
            # 4. Store Results
            results[f"{leg_key}ScanScore"] = score
            results[f"{leg_key}Notes"] = notes
            results[f"{leg_key}Quality"] = quality
            
            # Log Detailed Results for Validation
            print(f"📊 Leg Result [{leg_key}]:")
            print(f"   - pastern_angle: {p_angle:.2f}")
            print(f"   - hoof_angle: {h_angle:.2f}")
            print(f"   - hpa_dev: {prediction.get('hpa_dev'):.2f}")
            print(f"   - model_confidence: {conf:.2f}")
            print(f"   - calculated_score: {score}")
            
            leg_scores.append(score)

        except Exception as e:
            print(f"❌ Error during inference for {leg_key}: {traceback.format_exc()}")
            results[f"{leg_key}ScanScore"] = None
            results[f"{leg_key}Notes"] = f"Inference error: {str(e)}"
            results[f"{leg_key}Quality"] = None

    # 3. Aggregate
    aggregation = aggregate_scan(leg_scores)
    
    return AdvancedScanResponse(
        **results,
        scanScore=aggregation["scanScore"],
        notes=aggregation["notes"],
        quality=1,      # Static for now
        #status=1,       # Static for now
        #scanId="#SCN001" # Static for now
    )
