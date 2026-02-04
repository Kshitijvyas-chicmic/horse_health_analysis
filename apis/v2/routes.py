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

    results = {}
    leg_scores = []

    for leg_key, image_input in legs.items():
        if not image_input: # Handles None and empty string ""
            results[f"{leg_key}ScanScore"] = None
            results[f"{leg_key}Notes"] = None
            results[f"{leg_key}Quality"] = None
            continue

        try:
            # 1. Get Image
            img_bytes = await get_image_bytes(image_input)
            
            # 2. Inference
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
            
            # Log Results for Validation
            print(f"📊 Leg Result [{leg_key}]:")
            print(f"   - pastern_angle: {p_angle:.2f}")
            print(f"   - hoof_angle: {h_angle:.2f}")
            print(f"   - hpa_dev: {prediction.get('hpa_dev'):.2f}")
            print(f"   - model_confidence: {conf:.2f}")
            print(f"   - calculated_score: {score}")
            
            leg_scores.append(score)

        except Exception as e:
            print(f"❌ Error analyzing {leg_key}: {traceback.format_exc()}")
            results[f"{leg_key}ScanScore"] = None
            results[f"{leg_key}Notes"] = f"Error: {str(e)}"
            results[f"{leg_key}Quality"] = None

    # 5. Aggregate Overall
    aggregation = aggregate_scan(leg_scores)
    
    return AdvancedScanResponse(
        **results,
        scanScore=aggregation["scanScore"],
        notes=aggregation["notes"],
        quality=1,      # Static for now
        #status=1,       # Static for now
        #scanId="#SCN001" # Static for now
    )
