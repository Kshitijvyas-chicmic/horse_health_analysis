from fastapi import APIRouter, HTTPException, Request
from apis.schemas import BatchAnalysisRequest, BatchAnalysisResponse, ImageAnalysisResponse, AnalysisMetrics
import httpx
import asyncio
import traceback

router = APIRouter()

async def fetch_image(client: httpx.AsyncClient, url: str):
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"❌ Failed to fetch image from {url}: {e}")
        return None

@router.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze(request: BatchAnalysisRequest, req: Request):
    predictor = req.app.state.predictor
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not initialized")

    results = []
    async with httpx.AsyncClient() as client:
        # Step 1: Fetch all images concurrently to save time on network I/O
        fetch_tasks = [fetch_image(client, img.url) for img in request.images]
        image_contents = await asyncio.gather(*fetch_tasks)

        # Step 2: Process images one by one to prevent model/memory overload
        for i, img_req in enumerate(request.images):
            img_content = image_contents[i]
            
            if img_content is None:
                metrics = AnalysisMetrics(success=False, error="Failed to fetch image from URL")
            else:
                try:
                    # Sequential processing
                    prediction = predictor.predict(img_content)
                    metrics = AnalysisMetrics(**prediction)
                    
                    # Log Results for Validation
                    print(f"📊 Batch Item Result [{img_req.image_id}]:")
                    print(f"   - pastern_angle: {prediction.get('pastern_angle')}")
                    print(f"   - hoof_angle: {prediction.get('hoof_angle')}")
                    print(f"   - hpa_dev: {prediction.get('hpa_dev')}")
                    print(f"   - model_confidence: {prediction.get('model_confidence')}")
                    
                except Exception as e:
                    print(f"❌ Error processing image {img_req.image_id}: {traceback.format_exc()}")
                    metrics = AnalysisMetrics(success=False, error=str(e))
            
            results.append(ImageAnalysisResponse(image_id=img_req.image_id, metrics=metrics))

    return BatchAnalysisResponse(results=results)
