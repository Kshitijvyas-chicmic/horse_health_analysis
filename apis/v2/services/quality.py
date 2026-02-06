def map_quality(confidence: float) -> int:
    """
    Converts model confidence (0.0 -> 1.0) to a quality score (1 -> 10).
    Formula: quality = round(confidence * 10), clamped 1-10.
    """
    quality = round(confidence * 10)
    return max(1, min(10, quality))

def validate_image_quality(keypoints: list, scores: list) -> tuple:
    """
    Validates image quality based on model confidence and anatomical sanity.
    
    Returns:
        (bool, str): (is_valid, error_reason)
    """
    import numpy as np
    
    # 1. Confidence Check
    # Strict threshold for quality API
    if not all(s > 0.40 for s in scores):
        return False, "Low Confidence (<0.4)"
        
    # 2. Anatomy Check
    p0, p1, p2, p3 = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    
    # Gravity Check: Vertical Ordering (P0 < P2 < P3)
    # Allow small margin of error (e.g. 10px) for slight tilts
    if p0[1] > p2[1] - 10: return False, "Fetlock below Coronary Band"
    if p2[1] > p3[1] - 10: return False, "Coronary Band below Toe"
    
    # Segment Length Ratio Check
    def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))
    
    pastern_len = dist(p0, p1)
    hoof_wall_len = dist(p2, p3)
    
    # Safety checks for localized clusters
    if pastern_len < 10 or hoof_wall_len < 10:
        return False, "Keypoints too Clustered"

    ratio = pastern_len / hoof_wall_len
    
    if ratio > 5.0: return False, "Pastern disproportionately long"
    if ratio < 0.2: return False, "Hoof disproportionately long"
    
    return True, "OK"
