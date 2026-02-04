from typing import List, Optional

def aggregate_scan(leg_scores: List[float]) -> dict:
    """
    Aggregates per-leg scores into an overall result.
    
    Formula:
    scanScore = average(all available leg scores)
    Rounded to 1 decimal place.
    
    Avg Score   Category
    7 – 10      Healthy
    < 7 – ≥ 5   Mild concern
    < 5 – > 2   Needs attention
    ≤ 2         Critical
    """
    if not leg_scores:
        return {
            "scanScore": None,
            "notes": "No legs analyzed."
        }
        
    avg_score = sum(leg_scores) / len(leg_scores)
    avg_score = round(avg_score, 1)
    
    if avg_score >= 7:
        note = "Healthy"
    elif avg_score >= 5:
        note = "Mild concern"
    elif avg_score >= 2:
        note = "Needs attention"
    else:
        note = "Critical"
        
    return {
        "scanScore": avg_score,
        "notes": note
    }
