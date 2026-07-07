from pydantic import BaseModel
from typing import Optional
from apis.v3.schemas import AdvancedScanResponse, ModelResult


class ModelResultV5(ModelResult):
    """
    V5 extension of ModelResult.
    Adds lateral image URLs — annotated original lateral images uploaded to S3.
    """
    # --- Lateral Image URLs (annotated originals) ---
    frontLeftImageUrl: Optional[str] = None
    frontRightImageUrl: Optional[str] = None
    backLeftImageUrl: Optional[str] = None
    backRightImageUrl: Optional[str] = None


class AdvancedScanResponseV5(AdvancedScanResponse):
    """
    V5 extension of AdvancedScanResponse.
    Uses ModelResultV5 so that lateral image URL fields are included
    in the serialized API response (FastAPI uses the declared type for output).
    """
    mmpose: Optional[ModelResultV5] = None
    yolo: Optional[ModelResultV5] = None


class AdvancedScanRequest(BaseModel):
    # --- Lateral Views (Original with background) ---
    frontLeftLateral: Optional[str] = None
    frontRightLateral: Optional[str] = None
    backLeftLateral: Optional[str] = None
    backRightLateral: Optional[str] = None

    # --- Lateral Views (Background-removed, used for AI inference) ---
    frontLeftLateralProcessed: Optional[str] = None
    frontRightLateralProcessed: Optional[str] = None
    backLeftLateralProcessed: Optional[str] = None
    backRightLateralProcessed: Optional[str] = None

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
