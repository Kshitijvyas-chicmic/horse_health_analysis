import math

def calculate_leg_score(pastern_angle: float, hoof_angle: float) -> float:
    """
    Calculates the score based on the deviation between pastern and hoof angles.
    
    Rule:
    - Start score: 10
    - Positive dev (pastern > hoof): 1 degree -> -0.5 score
    - Negative dev (pastern < hoof): 1.5 degree -> -0.5 score
    - Clamp [1.0, 10.0]
    - Steps of 0.5
    """
    diff = pastern_angle - hoof_angle
    
    score = 10.0
    
    if diff >= 0:
        penalty_steps = diff / 1.0
    else:
        penalty_steps = abs(diff) / 1.5
        
    score = 10.0 - (penalty_steps * 0.5)
    
    # Quantize to 0.5 steps
    # Note: Using 0.5 steps as requested. Round to nearest 0.5.
    score = round(score * 2) / 2
    
    # Clamp
    return max(1.0, min(10.0, score))
