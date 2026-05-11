from pydantic import BaseModel
from typing import Optional


class AdvancedScanRequest(BaseModel):
    frontLeftLateral: Optional[str] = None
    frontRightLateral: Optional[str] = None
    backLeftLateral: Optional[str] = None
    backRightLateral: Optional[str] = None
    # --- Frontal Views ---
    frontLeftFrontal: Optional[str] = None
    frontRightFrontal: Optional[str] = None
    backLeftFrontal: Optional[str] = None
    backRightFrontal: Optional[str] = None


class ModelResult(BaseModel):
    """
    Flat per-leg results for a SINGLE model (either MMPose or YOLO).
    Field names are IDENTICAL between both models so the frontend can
    switch by reading response['mmpose'] vs response['yolo'] — no other
    code change needed.
    """

    # ── Per-leg Scan Scores ──────────────────────────────────
    frontLeftScanScore: Optional[float] = None
    frontRightScanScore: Optional[float] = None
    backLeftScanScore: Optional[float] = None
    backRightScanScore: Optional[float] = None
    # --- Frontal Scan Scores ---
    frontLeftFrontalScanScore: Optional[float] = None
    frontRightFrontalScanScore: Optional[float] = None
    backLeftFrontalScanScore: Optional[float] = None
    backRightFrontalScanScore: Optional[float] = None

    # ── Per-leg Granular Clinical Angles ─────────────────────
    frontLeftHoofAngle: Optional[float] = None
    frontLeftPasternAngle: Optional[float] = None
    frontLeftAngleDeviation: Optional[float] = None

    frontRightHoofAngle: Optional[float] = None
    frontRightPasternAngle: Optional[float] = None
    frontRightAngleDeviation: Optional[float] = None

    backLeftHoofAngle: Optional[float] = None
    backLeftPasternAngle: Optional[float] = None
    backLeftAngleDeviation: Optional[float] = None

    backRightHoofAngle: Optional[float] = None
    backRightPasternAngle: Optional[float] = None
    backRightAngleDeviation: Optional[float] = None

    # --- Frontal Granular Clinical Angles ---
    frontLeftFrontalHoofAngle: Optional[float] = None
    frontLeftFrontalPasternAngle: Optional[float] = None
    frontLeftFrontalAngleDeviation: Optional[float] = None

    frontRightFrontalHoofAngle: Optional[float] = None
    frontRightFrontalPasternAngle: Optional[float] = None
    frontRightFrontalAngleDeviation: Optional[float] = None

    backLeftFrontalHoofAngle: Optional[float] = None
    backLeftFrontalPasternAngle: Optional[float] = None
    backLeftFrontalAngleDeviation: Optional[float] = None

    backRightFrontalHoofAngle: Optional[float] = None
    backRightFrontalPasternAngle: Optional[float] = None
    backRightFrontalAngleDeviation: Optional[float] = None

    # ── Per-leg Clinical Notes ───────────────────────────────
    frontLeftNotes: Optional[str] = None
    frontRightNotes: Optional[str] = None
    backLeftNotes: Optional[str] = None
    backRightNotes: Optional[str] = None
    # --- Frontal Notes ---
    frontLeftFrontalNotes: Optional[str] = None
    frontRightFrontalNotes: Optional[str] = None
    backLeftFrontalNotes: Optional[str] = None
    backRightFrontalNotes: Optional[str] = None

    # ── Per-leg Condition ────────────────────────────────────
    frontLeftCondition: Optional[str] = None
    frontRightCondition: Optional[str] = None
    backLeftCondition: Optional[str] = None
    backRightCondition: Optional[str] = None
    # --- Frontal Condition ---
    frontLeftFrontalCondition: Optional[str] = None
    frontRightFrontalCondition: Optional[str] = None
    backLeftFrontalCondition: Optional[str] = None
    backRightFrontalCondition: Optional[str] = None

    # ── Per-leg Recommendation ───────────────────────────────
    frontLeftRecommendation: Optional[str] = None
    frontRightRecommendation: Optional[str] = None
    backLeftRecommendation: Optional[str] = None
    backRightRecommendation: Optional[str] = None
    # --- Frontal Recommendation ---
    frontLeftFrontalRecommendation: Optional[str] = None
    frontRightFrontalRecommendation: Optional[str] = None
    backLeftFrontalRecommendation: Optional[str] = None
    backRightFrontalRecommendation: Optional[str] = None

    # ── Per-leg Quality Check (Pass / Fail) ──────────────────
    frontLeftQualityCheck: Optional[str] = None
    frontRightQualityCheck: Optional[str] = None
    backLeftQualityCheck: Optional[str] = None
    backRightQualityCheck: Optional[str] = None
    # --- Frontal Quality Check ---
    frontLeftFrontalQualityCheck: Optional[str] = None
    frontRightFrontalQualityCheck: Optional[str] = None
    backLeftFrontalQualityCheck: Optional[str] = None
    backRightFrontalQualityCheck: Optional[str] = None

    # ── Per-leg Quality Score (int) ──────────────────────────
    frontLeftQuality: Optional[int] = None
    frontRightQuality: Optional[int] = None
    backLeftQuality: Optional[int] = None
    backRightQuality: Optional[int] = None
    # --- Frontal Quality Score ---
    frontLeftFrontalQuality: Optional[int] = None
    frontRightFrontalQuality: Optional[int] = None
    backLeftFrontalQuality: Optional[int] = None
    backRightFrontalQuality: Optional[int] = None


class AdvancedScanResponse(BaseModel):
    """
    Dual-model V3 response.
    Both 'mmpose' and 'yolo' blocks have identical field names.
    Frontend switches model by reading response['mmpose'] or response['yolo'].

    Overall scanScore is derived from MMPose as the primary clinical source.
    """
    mmpose: Optional[ModelResult] = None
    yolo: Optional[ModelResult] = None

    # Overall aggregate (MMPose as primary)
    scanScore: Optional[float] = None
    quality: int = 1
    notes: str
