from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import logging

# Ensure mmpose is in path for logic imports
sys.path.append('mmpose')

from apis.logic import HPAPredictor
from apis.yolo_predictor import YOLOPredictor
from apis.routes.v1.analyze import router as analyze_router
from apis.v2.routes import router as analyze_v2_router
from apis.v3.routes import router as analyze_v3_router

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Absolute Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Horse Health Analysis API",
    description="API for detecting horse hoof and pastern keypoints and calculating HPA metrics.",
    version="1.0.2",
    servers=[
        {"url": "https://horse-health.projectlabs.in", "description": "Production server"},
        {"url": "https://horse-health-new.projectlabs.in", "description": "Production new server"},
        {"url": "http://192.180.3.178:8001", "description": "Test server for sharing"}
    ],
    redirect_slashes=False  # CRITICAL: Prevent 307 redirects which break CORS preflights
)

# 1. Proxy Support (Inner Middleware)
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# 2. Request Logging
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"📡 Request: {request.method} {request.url}")
    return await call_next(request)

# 3. CORS Middleware (Outer Middleware - Added LAST)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Configuration — Absolute Paths for Deployment
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'mmpose/work_dirs/rtmpose_hoof_manual_9_march/rtmpose_hoof_4kp_copy.py')
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'mmpose/work_dirs/rtmpose_hoof_manual_9_march/epoch_300.pth')
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, 'runs/segment/hpa_v8m_full_v1/weights/best.pt')
DEVICE = 'cpu'

# Initialize predictors at startup (loaded once, reused across all requests)
@app.on_event("startup")
async def startup_event():
    # 1. MMPose
    logger.info("🚀 Initializing HPAPredictor (MMPose)...")
    try:
        app.state.predictor = HPAPredictor(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
        logger.info("✅ HPAPredictor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize HPAPredictor: {e}")
        app.state.predictor = None

    # 2. YOLO Medium
    logger.info("🚀 Initializing YOLOPredictor (YOLO Medium)...")
    try:
        if not os.path.exists(YOLO_WEIGHTS):
            logger.error(f"❌ YOLO weights not found at: {YOLO_WEIGHTS}")
            app.state.yolo_predictor = None
        else:
            app.state.yolo_predictor = YOLOPredictor(YOLO_WEIGHTS)
            logger.info("✅ YOLOPredictor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize YOLOPredictor: {e}")
        app.state.yolo_predictor = None

@app.get("/api/status", tags=["Health"])
async def status():
    """Returns the initialization status of the models."""
    return {
        "mmpose": "loaded" if getattr(app.state, "predictor", None) else "failed",
        "yolo": "loaded" if getattr(app.state, "yolo_predictor", None) else "failed",
        "paths": {
            "root": PROJECT_ROOT,
            "yolo_exists": os.path.exists(YOLO_WEIGHTS)
        }
    }

@app.get("/", tags=["Health"])
@app.get("/ping", tags=["Health"])
async def root():
    return {
        "message": "Horse Health Analysis API is running", 
        "status": "stable", 
        "version": "1.0.2",
        "last_updated": "2026-02-05 16:15"
    }

# Include versioned routers
app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis V1"])
app.include_router(analyze_v2_router, prefix="/api/v2", tags=["Analysis V2 (MMPose)"])
app.include_router(analyze_v3_router, prefix="/api/v3", tags=["Analysis V3 (Dual Model)"])

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
        logger.error(f"❌ Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, proxy_headers=True, forwarded_allow_ips="*")
