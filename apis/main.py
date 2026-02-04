from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Ensure mmpose is in path for logic imports
sys.path.append('mmpose')

from apis.logic import HPAPredictor
from apis.routes.v1.analyze import router as analyze_router
from apis.v2.routes import router as analyze_v2_router

app = FastAPI(
    title="Horse Health Analysis API",
    description="API for detecting horse hoof and pastern keypoints and calculating HPA metrics.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG_PATH = 'mmpose/work_dirs/rtmpose_hoof_unified_jan12/rtmpose_hoof_unified_jan12.py'
CHECKPOINT_PATH = 'mmpose/work_dirs/rtmpose_hoof_unified_jan12/epoch_300.pth'
DEVICE = 'cpu'

# Initialize predictor at startup and store in app state
@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing HPAPredictor...")
    try:
        app.state.predictor = HPAPredictor(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
        print("✅ HPAPredictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize HPAPredictor: {e}")
        app.state.predictor = None

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Horse Health Analysis API is running", "status": "stable", "version": "1.0.0"}

# Include versioned routers
app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis V1"])
app.include_router(analyze_v2_router, prefix="/api/v2", tags=["Analysis V2"])

# Keep the legacy endpoint for backward compatibility if needed, or remove it
@app.post("/analyze", tags=["Legacy"])
async def analyze_image(file: UploadFile = File(...)):
    predictor = getattr(app.state, 'predictor', None)
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        results = predictor.predict(contents)
        
        # Log Results for Validation
        print(f"📊 Legacy API Result [{file.filename}]:")
        print(f"   - pastern_angle: {results.get('pastern_angle')}")
        print(f"   - hoof_angle: {results.get('hoof_angle')}")
        print(f"   - hpa_dev: {results.get('hpa_dev')}")
        print(f"   - model_confidence: {results.get('model_confidence')}")
        
        return JSONResponse(content=results)
    except Exception as e:
        import traceback
        print(f"❌ Error during analysis: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
