from pydantic import BaseModel
from typing import Optional
from apis.v3.schemas import AdvancedScanResponse, ModelResult

class AdvancedScanRequest(BaseModel):
    frontLeftLateral: Optional[str] = None
    frontRightLateral: Optional[str] = None
    backLeftLateral: Optional[str] = None
    backRightLateral: Optional[str] = None
    
    # --- Frontal Views (Original with background) ---
    frontLeftFrontal: Optional[str] = None
    frontRightFrontal: Optional[str] = None
    backLeftFrontal: Optional[str] = None
    backRightFrontal: Optional[str] = None

    # --- Frontal Views (Processed/Background removed) ---
    frontLeftFrontalProcessed: Optional[str] = None
    frontRightFrontalProcessed: Optional[str] = None
    backLeftFrontalProcessed: Optional[str] = None
    backRightFrontalProcessed: Optional[str] = None
