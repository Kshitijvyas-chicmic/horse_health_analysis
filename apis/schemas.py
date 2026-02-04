from pydantic import BaseModel
from typing import List, Optional

class ImageRequest(BaseModel):
    image_id: str
    url: str

class BatchAnalysisRequest(BaseModel):
    images: List[ImageRequest]

class AnalysisMetrics(BaseModel):
    success: bool
    best_zone: Optional[str] = None
    pastern_angle: Optional[float] = None
    hoof_angle: Optional[float] = None
    hpa_dev: Optional[float] = None
    model_confidence: Optional[float] = None
    image_base64: Optional[str] = None
    error: Optional[str] = None

class ImageAnalysisResponse(BaseModel):
    image_id: str
    metrics: AnalysisMetrics

class BatchAnalysisResponse(BaseModel):
    results: List[ImageAnalysisResponse]
