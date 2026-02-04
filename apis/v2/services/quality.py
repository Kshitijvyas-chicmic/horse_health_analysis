def map_quality(confidence: float) -> int:
    """
    Converts model confidence (0.0 -> 1.0) to a quality score (1 -> 10).
    
    Formula:
    quality = round(confidence * 10)
    Clamp between 1 and 10
    """
    quality = round(confidence * 10)
    return max(1, min(10, quality))
