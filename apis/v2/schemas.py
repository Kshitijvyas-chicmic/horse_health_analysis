from pydantic import BaseModel
from typing import List, Optional

class AdvancedScanRequest(BaseModel):
    frontLeftLateral: Optional[str] = None
    frontRightLateral: Optional[str] = None
    backLeftLateral: Optional[str] = None
    backRightLateral: Optional[str] = None

class AdvancedScanResponse(BaseModel):
    # Per-leg Fields
    frontLeftScanScore: Optional[float] = None
    frontRightScanScore: Optional[float] = None
    backLeftScanScore: Optional[float] = None
    backRightScanScore: Optional[float] = None

    frontLeftNotes: Optional[str] = None
    frontRightNotes: Optional[str] = None
    backLeftNotes: Optional[str] = None
    backRightNotes: Optional[str] = None

    frontLeftCondition: Optional[str] = None
    frontRightCondition: Optional[str] = None
    backLeftCondition: Optional[str] = None
    backRightCondition: Optional[str] = None

    frontLeftRecommendation: Optional[str] = None
    frontRightRecommendation: Optional[str] = None
    backLeftRecommendation: Optional[str] = None
    backRightRecommendation: Optional[str] = None

    # Quality Check Fields (Pass/Fail)
    frontLeftQualityCheck: Optional[str] = None
    frontRightQualityCheck: Optional[str] = None
    backLeftQualityCheck: Optional[str] = None
    backRightQualityCheck: Optional[str] = None

    frontLeftQuality: Optional[int] = None
    frontRightQuality: Optional[int] = None
    backLeftQuality: Optional[int] = None
    backRightQuality: Optional[int] = None

    # Overall Scan Fields
    scanScore: Optional[float] = None
    quality: int = 1
    notes: str
    #status: int = 1
    #scanId: str = "#SCN001"
