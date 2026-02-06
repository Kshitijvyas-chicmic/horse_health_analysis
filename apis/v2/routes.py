from fastapi import APIRouter, Request, HTTPException
from apis.v2.schemas import AdvancedScanRequest, AdvancedScanResponse
from apis.v2.services.inference import get_image_bytes, run_leg_inference
from apis.v2.services.scoring import calculate_leg_score
from apis.v2.services.notes import map_leg_notes
from apis.v2.services.quality import map_quality
from apis.v2.services.aggregator import aggregate_scan
from apis.v2.services.clinical import map_condition, map_clinical_notes, map_recommendation
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
            results[f"{leg_key}Condition"] = None
            results[f"{leg_key}Recommendation"] = None
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
            results[f"{leg_key}Condition"] = None
            results[f"{leg_key}Recommendation"] = None
            results[f"{leg_key}Quality"] = None
        else:
            leg_data[leg_key] = result

    # 2. Parallel Inference (Using ThreadPool to utilize multiple cores if GIL is released)
    print(f"🧠 Running inference on {len(leg_data)} legs in parallel...")
    from concurrent.futures import ThreadPoolExecutor
    
    def process_single_leg(leg_key, img_bytes):
        try:
            prediction = run_leg_inference(predictor, img_bytes)
            return leg_key, prediction
        except Exception as e:
            return leg_key, {"success": False, "error": str(e)}

    # We use a loop to create thread tasks
    # Note: Torch usually releases the GIL during heavy math, so this should scale!
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        inference_tasks = [
            loop.run_in_executor(pool, process_single_leg, k, b) 
            for k, b in leg_data.items()
        ]
        inference_results = await asyncio.gather(*inference_tasks)

    # 3. Process results
    leg_scores = []
    for leg_key, prediction in inference_results:
        if not prediction.get("success"):
            print(f"❌ Inference failed for {leg_key}: {prediction.get('error')}")
            results[f"{leg_key}ScanScore"] = None
            results[f"{leg_key}Notes"] = f"Inference failed: {prediction.get('error')}"
            results[f"{leg_key}Condition"] = None
            results[f"{leg_key}Recommendation"] = None
            results[f"{leg_key}Quality"] = None
            continue

        # Calculate scores and clinical details
        p_angle = prediction["pastern_angle"]
        h_angle = prediction["hoof_angle"]
        conf = prediction["model_confidence"]
        
        score = calculate_leg_score(p_angle, h_angle)
        quality = map_quality(conf)
        notes = map_clinical_notes(score)  # Using new detailed notes
        condition = map_condition(score)
        recommendation = map_recommendation(score)
        
        # Store Results
        results[f"{leg_key}ScanScore"] = score
        results[f"{leg_key}Notes"] = notes
        results[f"{leg_key}Condition"] = condition
        results[f"{leg_key}Recommendation"] = recommendation
        results[f"{leg_key}Quality"] = quality
        
        # Log Detailed Results
        print(f"✅ Analyzed {leg_key}: Score={score}, Condition={condition}")
        leg_scores.append(score)

    # 4. Aggregate
    aggregation = aggregate_scan(leg_scores)
    
    return AdvancedScanResponse(
        **results,
        scanScore=aggregation["scanScore"],
        notes=aggregation["notes"],
        quality=1,      # Static for now
        #status=1,       # Static for now
        #scanId="#SCN001" # Static for now
    )
